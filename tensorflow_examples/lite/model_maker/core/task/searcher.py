# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""APIs to train a model that can search Searcher Task."""

import dataclasses
import enum
import logging
import os
import tempfile
from typing import AnyStr, List, Optional

import tensorflow as tf
import flatbuffers
from tensorflow_lite_support.metadata import metadata_schema_py_generated as _metadata_fb
from tensorflow_lite_support.metadata import schema_py_generated as _schema_fb
from tensorflow_lite_support.metadata.python import metadata as _metadata
from tensorflow_examples.lite.model_maker.core.data_util import searcher_dataloader
from tensorflow_examples.lite.model_maker.core.utils import ondevice_scann_builder
from tensorflow_examples.lite.model_maker.core.utils import scann_converter


@enum.unique
class ExportFormat(enum.Enum):
  TFLITE = "TFLITE"
  SCANN_INDEX_FILE = "SCANN_INDEX_FILE"


@dataclasses.dataclass
class Tree:
  """K-Means partitioning tree configuration.

  In ScaNN, we use single layer K-Means tree to partition the database (index)
  as a way to reduce search space.

  Attributes:
    num_leaves: How many leaves (partitions) to have on the K-Means tree. In
      general, a good starting point would be the square root of the database
      size.
    num_leaves_to_search: During inference ScaNN will compare the query vector
      against all the partition centroids and select the closest
      `num_leaves_to_search` ones to search in. The more leaves to search, the
      better the retrieval quality, and higher computational cost.
    training_sample_size: How many database embeddings to sample for the K-Means
      training. Generally, you want to use a large enough sample of the database
      to train K-Means so that it's representative enough. However, large sample
      can also lead to longer training time. A good starting value would be
      100k, or the whole dataset if it's smaller than that.
    min_partition_size: Smallest allowable cluster size. Any clusters smaller
      than this will be removed, and its data points will be merged with other
      clusters. Recommended to be 1/10 of average cluster size (size of database
      divided by `num_leaves`)
    training_iterations: How many itrations to train K-Means.
    spherical: If true, L2 normalize the K-Means centroids.
    quantize_centroids: If true, quantize centroids to int8.
    random_init: If true, use random init. Otherwise use K-Means++.
  """
  num_leaves: int
  num_leaves_to_search: int
  training_sample_size: int = 100000
  min_partition_size: int = 50
  training_iterations: int = 12
  spherical: bool = False
  quantize_centroids: bool = False
  random_init: bool = True


@dataclasses.dataclass
class ScoreAH:
  """Product Quantization (PQ) based in-partition scoring configuration.

  In ScaNN we use PQ to compress the database embeddings, but not the query
  embedding. We called it Asymmetric Hashing. See
  https://research.google/pubs/pub41694/

  Attributes:
    dimensions_per_block: How many dimensions in each PQ block. If the embedding
      vector dimensionality is a multiple of this value, there will be
      `number_of_dimensions / dimensions_per_block` PQ blocks. Otherwise, the
      last block will be the remainder. For example, if a vector has 12
      dimensions, and `dimensions_per_block` is 2, then there will be 6
      2-dimension blocks. However, if the vector has 13 dimensions and
      `dimensions_per_block` is still 2, there will be 6 2-dimension blocks and
      one 1-dimension block.
    anisotropic_quantization_threshold: If this value is set, we will penalize
      the quantization error that's parallel to the original vector differently
      than the orthogonal error. A generally recommended value for this
      parameter would be 0.2. For more details, please look at ScaNN's 2020 ICML
      paper https://arxiv.org/abs/1908.10396 and the Google AI Blog post
      https://ai.googleblog.com/2020/07/announcing-scann-efficient-vector.html
    training_sample_size: How many database points to sample for training the
      K-Means for PQ centers. A good starting value would be 100k or the whole
      dataset if it's smaller than that.
    training_iterations: How many iterations to run K-Means for PQ.
  """
  dimensions_per_block: int
  anisotropic_quantization_threshold: float = float("nan")
  training_sample_size: int = 100000

  training_iterations: int = 10


@dataclasses.dataclass
class ScoreBruteForce:
  """Bruce force in-partition scoring configuration.

  There'll be no compression or quantization applied to the database
  embeddings or query embeddings.
  """


@dataclasses.dataclass
class ScaNNOptions:
  """Options to build ScaNN.

  ScaNN
  (https://ai.googleblog.com/2020/07/announcing-scann-efficient-vector.html) is
  a highly efficient and scalable vector nearest neighbor retrieval
  library from Google Research. We use ScaNN to build the on-device search
  index, and do on-device retrieval with a simplified implementation.
  TODO(b/231134703) Add a link to the README

  Attributes:
    distance_measure: How to compute the distance. Allowed values are
      'dot_product' and 'squared_l2'. Please note that when distance is
      'dot_product', we actually compute the negative dot product between query
      and database vectors, to preserve the notion that "smaller is closer".
    tree: Configure partitioning. If not set, no partitioning is performed.
    score_ah: Configure asymmetric hashing. Must defined this or
      `score_brute_force`.
    score_brute_force: Configure bruce force. Must defined this or `score_ah`.
  """
  distance_measure: str
  tree: Optional[Tree] = None
  score_ah: Optional[ScoreAH] = None
  score_brute_force: Optional[ScoreBruteForce] = None


class Searcher(object):
  """Creates the similarity search model with ScaNN."""

  def __init__(self,
               serialized_scann_path: str,
               metadata: List[AnyStr],
               embedder_path: Optional[str] = None) -> None:
    """Initializes the Searcher object.

    Args:
      serialized_scann_path: Path to the dir that contains the ScaNN's
        artifacts.
      metadata: The metadata for each of the embeddings in the database. Passed
        in the same order as the embeddings in ScaNN.
      embedder_path: Path to the TFLite Embedder model file.
    """
    self._serialized_scann_path = serialized_scann_path
    self._metadata = metadata
    self._embedder_path = embedder_path

  @classmethod
  def create_from_server_scann(
      cls,
      serialized_scann_path: str,
      metadata: List[AnyStr],
      embedder_path: Optional[str] = None) -> "Searcher":
    """Creates the instance from the serialized serving scann directory.

    Args:
      serialized_scann_path: Path to the dir that contains the ScaNN's
        artifacts.
      metadata: The metadata for each of the embeddings in the database. Passed
        in the same order as the embeddings in ScaNN.
      embedder_path: Path to the TFLite Embedder model file.

    Returns:
      A Searcher instance.
    """
    return cls(serialized_scann_path, metadata, embedder_path)

  @classmethod
  def create_from_data(cls,
                       data: searcher_dataloader.DataLoader,
                       scann_options: ScaNNOptions,
                       cache_dir: Optional[str] = None) -> "Searcher":
    """"Creates the instance from data.

    Args:
      data: Data used to create scann.
      scann_options: Options to build the ScaNN index file.
      cache_dir: The cache directory to save serialized ScaNN and/or the tflite
        model. When cache_dir is not set, a temporary folder will be created and
        will **not** be removed automatically which makes it can be used later.

    Returns:
      A Searcher instance.
    """
    # Gets the ScaNN builder.
    builder = ondevice_scann_builder.builder(
        data.dataset,
        num_neighbors=10,  # This parameter is not used in on-device.
        distance_measure=scann_options.distance_measure)
    if scann_options.tree:
      builder = builder.tree(**dataclasses.asdict(scann_options.tree))
    if scann_options.score_ah:
      # We only support LUT256 for on-device.
      builder = builder.score_ah(
          hash_type="lut256", **dataclasses.asdict(scann_options.score_ah))
    if scann_options.score_brute_force:
      builder = builder.score_brute_force(
          **dataclasses.asdict(scann_options.score_brute_force))

    if cache_dir is None:
      cache_dir = tempfile.mkdtemp()
    if not tf.io.gfile.exists(cache_dir):
      tf.io.gfile.makedirs(cache_dir)
    logging.info("Cache will be stored in %s", cache_dir)

    # Builds, serializes and saves the ScaNN model.
    scann = builder.build()
    serialized_scann_path = os.path.join(cache_dir, "serialized_scann")
    if not tf.io.gfile.exists(serialized_scann_path):
      tf.io.gfile.makedirs(serialized_scann_path)
    scann.serialize(serialized_scann_path)

    return cls(serialized_scann_path, data.metadata, data.embedder_path)

  def export(self,
             export_format: ExportFormat,
             export_filename: str,
             userinfo: AnyStr,
             compression: bool = True):
    """Export the searcher model.

    Args:
      export_format: Export format that could be tflite or on-device ScaNN index
        file, must be `ExportFormat.TFLITE` or `ExportFormat.SCANN_INDEX_FILE`.
      export_filename: File name to save the exported file. The exported file
        can be TFLite model or on-device ScaNN index file.
      userinfo: A special field in the index file that can be an arbitrary
        string supplied by the user.
      compression: Whether to snappy compress the index file.
    """
    export_dir = os.path.dirname(export_filename)
    if not tf.io.gfile.exists(export_dir):
      tf.io.gfile.makedirs(export_dir)

    if export_format is ExportFormat.SCANN_INDEX_FILE:
      output_scann_path = export_filename
    elif export_format is ExportFormat.TFLITE:
      tmpdir = tempfile.mkdtemp()
      output_scann_path = os.path.join(tmpdir, "on_device_scann_index.ldb")
    else:
      raise ValueError("Unsupported export format: ", export_format)

    # Creates the on-device ScaNN index file and saves it in output_scann_path.
    artifacts = scann_converter.convert_serialized_to_on_device(
        self._serialized_scann_path)
    scann_converter.convert_artifacts_to_leveldb(
        output_scann_path,
        metadata=self._metadata,
        artifacts=artifacts,
        userinfo=userinfo,
        compression=compression)

    # Associates the scann index file with the tflite model file.
    if export_format is ExportFormat.TFLITE:
      # Creates the metadata populator.
      if self._embedder_path is None:
        raise ValueError("Can't export the tflite model since embedder model "
                         "is not provided.")
      with tf.io.gfile.GFile(self._embedder_path, "rb") as f:
        model_buffer = f.read()
        populator = _metadata.MetadataPopulator.with_model_buffer(model_buffer)

      # Extracts the metadata.
      metadata_buffer = _metadata.get_metadata_buffer(model_buffer)
      if metadata_buffer:
        metadata = _metadata_fb.ModelMetadataT.InitFromObj(
            _metadata_fb.ModelMetadata.GetRootAsModelMetadata(
                metadata_buffer, 0))
      else:
        # Creates the empty metadata.
        model = _schema_fb.Model.GetRootAsModel(model_buffer, 0)
        num_input_tensors = model.Subgraphs(0).InputsLength()
        input_metadata = [
            _metadata_fb.TensorMetadataT() for i in range(num_input_tensors)
        ]
        num_output_tensors = model.Subgraphs(0).OutputsLength()
        output_metadata = [
            _metadata_fb.TensorMetadataT() for i in range(num_output_tensors)
        ]

        subgraph_metadata = _metadata_fb.SubGraphMetadataT()
        subgraph_metadata.inputTensorMetadata = input_metadata
        subgraph_metadata.outputTensorMetadata = output_metadata

        metadata = _metadata_fb.ModelMetadataT()
        metadata.subgraphMetadata = [subgraph_metadata]

      # Updates the metadata with the scann associated file.
      scann_file = _metadata_fb.AssociatedFileT()
      scann_file.name = os.path.basename(output_scann_path)
      scann_file.description = "On-device Scann Index file with LevelDB format."
      scann_file.type = _metadata_fb.AssociatedFileType.SCANN_INDEX_FILE
      output_metadata = metadata.subgraphMetadata[0].outputTensorMetadata[0]
      if output_metadata.associatedFiles is None:
        output_metadata.associatedFiles = [scann_file]
      else:
        output_metadata.associatedFiles.append(scann_file)

      # Saves the updated metadata and the scann associated file along with the
      # model buffer.
      buffer_builder = flatbuffers.Builder(0)
      buffer_builder.Finish(
          metadata.Pack(buffer_builder),
          _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER)
      updated_metadata_buffer = buffer_builder.Output()
      populator.load_metadata_buffer(updated_metadata_buffer)
      populator.load_associated_files([output_scann_path])
      populator.populate()

      output_tflite_path = export_filename
      with tf.io.gfile.GFile(output_tflite_path, "wb") as f:
        f.write(populator.get_model_buffer())
      tf.io.gfile.rmtree(tmpdir)
