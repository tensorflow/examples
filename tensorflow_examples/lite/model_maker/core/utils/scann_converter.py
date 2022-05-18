# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Util for converting ScaNN artifacts to on-device format."""
import os
from typing import NamedTuple, Optional, List, AnyStr

import numpy as np
import tensorflow as tf

from tensorflow_lite_support.scann_ondevice.cc.core import serialized_searcher_pb2
from scann.data_format import features_pb2
from scann.partitioning import partitioner_pb2
from scann.proto import centers_pb2
from scann.proto import hash_pb2
from scann.proto import scann_pb2
from tensorflow_lite_support.scann_ondevice.cc.python import index_builder

_DISTANCE_MAP = {
    'SquaredL2Distance': serialized_searcher_pb2.SQUARED_L2_DISTANCE,
    'DotProductDistance': serialized_searcher_pb2.DOT_PRODUCT,
}
_LOOKUP_TYPE_MAP = {
    hash_pb2.AsymmetricHasherConfig.FLOAT:
        serialized_searcher_pb2.AsymmetricHashingProto.FLOAT,
    hash_pb2.AsymmetricHasherConfig.INT8:
        serialized_searcher_pb2.AsymmetricHashingProto.INT8,
    hash_pb2.AsymmetricHasherConfig.INT16:
        serialized_searcher_pb2.AsymmetricHashingProto.INT16,
    hash_pb2.AsymmetricHasherConfig.INT8_LUT16:
        serialized_searcher_pb2.AsymmetricHashingProto.INT8_LUT16,
}


class OnDeviceArtifacts(NamedTuple):
  ondevice_config: serialized_searcher_pb2.ScannOnDeviceConfig
  hashed_dataset: Optional[np.ndarray]
  float_dataset: Optional[np.ndarray]
  partition_assignments: Optional[np.ndarray]


def get_distance_measure(
    distance_measure_str: str) -> serialized_searcher_pb2.DistanceMeasure:
  """Maps a distance measure string to a DistanceMeasure proto."""
  return _DISTANCE_MAP.get(distance_measure_str,
                           serialized_searcher_pb2.UNSPECIFIED)


def get_indexer(
    on_device_distance: serialized_searcher_pb2.DistanceMeasure,
    lookup_type: hash_pb2.AsymmetricHasherConfig.LookupType,
    ah_codebook: centers_pb2.CentersForAllSubspaces
) -> serialized_searcher_pb2.IndexerProto:
  """Helper util that builds an indexer for `convert_artifacts_to_leveldb`.

  Args:
    on_device_distance: DistanceMeasure used in NN search.
    lookup_type: Type of lookup table used for asymmetric distance NN search.
    ah_codebook: ScaNN AsymmetricHashing hasher.

  Returns:
    An IndexerProto.

  Raises:
    ValueError: If a subspace center in `ah_codebook` uses a feature type other
    than double or float.
  """
  indexer = serialized_searcher_pb2.IndexerProto()
  od_ah_codebook = indexer.asymmetric_hashing
  od_ah_codebook.query_distance = on_device_distance
  od_ah_codebook.lookup_type = _LOOKUP_TYPE_MAP[lookup_type]
  for subspace in ah_codebook.subspace_centers:
    od_subspace = od_ah_codebook.subspace.add()
    for center in subspace.center:
      entry = od_subspace.entry.add()
      feature_type = center.feature_type
      if feature_type == features_pb2.GenericFeatureVector.FeatureType.FLOAT:
        entry.dimension.extend(center.feature_value_float)
      elif feature_type == features_pb2.GenericFeatureVector.FeatureType.DOUBLE:
        entry.dimension.extend(center.feature_value_double)
      else:
        raise ValueError(
            f'ah_codebook has unsupported feature_type {feature_type}')
  return indexer


def get_partitioner(
    on_device_distance: serialized_searcher_pb2.DistanceMeasure,
    search_fraction: float,
    partition_centroids: partitioner_pb2.SerializedPartitioner
) -> serialized_searcher_pb2.PartitionerProto:
  """Helper util that builds a partitioner for `convert_artifacts_to_leveldb`.

  Args:
    on_device_distance: DistanceMeasure used in NN search.
    search_fraction: Fraction of partitions searched.
    partition_centroids: ScaNN partitioner.

  Returns:
    A PartitionerProto.
  """
  od_partitioner = serialized_searcher_pb2.PartitionerProto()
  od_partitioner.query_distance = on_device_distance
  od_partitioner.search_fraction = search_fraction
  for i, center in enumerate(
      partition_centroids.kmeans.kmeans_tree.root.centers):
    assert partition_centroids.kmeans.kmeans_tree.root.children[i].leaf_id == i
    leaf = od_partitioner.leaf.add()
    leaf.dimension.extend(center.dimension)
  return od_partitioner


def convert_serialized_to_on_device(serialized_path: str) -> OnDeviceArtifacts:
  """Converts ScaNN's serialized artifacts to on-device format.

  Args:
    serialized_path: Path to the dir that contains the ScaNN's artifacts.

  Returns:
    A named tuple containing the on-device ScannOnDeviceConfig as well as the
    database (maybe compressed) and partition assignment (if partitioning is
    enabled) in numpy format.
  """
  config_path = os.path.join(serialized_path, 'scann_config.pb')

  # Load ScannConfig
  with tf.io.gfile.GFile(config_path, 'rb') as config_pb:
    scann_config = scann_pb2.ScannConfig.FromString(config_pb.read())

  if scann_config.HasField('exact_reordering'):
    raise ValueError('exact_reordering is not supported on-device')

  on_device_config = serialized_searcher_pb2.ScannOnDeviceConfig()
  on_device_distance = get_distance_measure(
      scann_config.distance_measure.distance_measure)
  on_device_config.query_distance = on_device_distance

  # Indexed dataset parts
  ah_quantized_dataset = None
  float_dataset = None
  partition_assignments = None

  # Load AH centers
  if scann_config.HasField('hash'):
    with tf.io.gfile.GFile(
        os.path.join(serialized_path, 'ah_codebook.pb'), 'rb') as ah_pb:
      ah_codebook = centers_pb2.CentersForAllSubspaces.FromString(ah_pb.read())
      on_device_config.indexer.CopyFrom(
          get_indexer(on_device_distance,
                      scann_config.hash.asymmetric_hash.lookup_type,
                      ah_codebook))
    ah_quantized_dataset = np.load(
        os.path.join(serialized_path, 'hashed_dataset.npy'))

  # Load partition centroids
  if scann_config.HasField('partitioning'):
    with tf.io.gfile.GFile(
        os.path.join(serialized_path, 'serialized_partitioner.pb'),
        'rb') as partition_centroid_pb:
      partition_centroids = partitioner_pb2.SerializedPartitioner.FromString(
          partition_centroid_pb.read())
      on_device_config.partitioner.CopyFrom(
          get_partitioner(
              on_device_distance,
              (scann_config.partitioning.query_spilling.max_spill_centers /
               scann_config.partitioning.num_children), partition_centroids))
    partition_assignments = np.load(
        os.path.join(serialized_path, 'datapoint_to_token.npy'))

  # Load brute force datasets
  if scann_config.brute_force.fixed_point.enabled:
    raise NotImplementedError(
        'Fixed point int8 quantization is not yet supported on-device')
  elif scann_config.HasField('brute_force'):
    float_dataset = np.load(os.path.join(serialized_path, 'dataset.npy'))

  return OnDeviceArtifacts(on_device_config, ah_quantized_dataset,
                           float_dataset, partition_assignments)


def convert_artifacts_to_leveldb(output_file_path: str,
                                 metadata: List[AnyStr],
                                 userinfo: AnyStr,
                                 artifacts: OnDeviceArtifacts,
                                 compression: bool = True) -> None:
  """Converts artifacts to the index file.

  Raises exception if the input is invalid or failed to create the file.

  Args:
    output_file_path: Path to the levelDB index file.
    metadata: The metadata for each of the embeddings in the database. Passed in
      the same order as the embeddings in artifacts.
    userinfo: A special field in the index file that can be an arbitrary string
      supplied by the user.
    artifacts: Artifacts parsed by the convert_serialized_to_on_device
    compression: Whether to snappy compress the index file.
  """
  hashed_dataset = None
  float_dataset = None
  if artifacts.hashed_dataset is not None:
    embedding_dim = int(artifacts.hashed_dataset.shape[1])
    hashed_dataset = artifacts.hashed_dataset.reshape((-1,))
  if artifacts.float_dataset is not None:
    embedding_dim = int(artifacts.float_dataset.shape[1])
    float_dataset = artifacts.float_dataset.reshape((-1,))

  if len(artifacts.partition_assignments.shape) != 1:
    raise ValueError('Partition assignment array has to be 1D')

  # Raises exception if both hashed_dataset and float_dataset are not
  # None, or both are None
  serialized_index_file = index_builder.create_serialized_index_file(
      embedding_dim,
      artifacts.ondevice_config.SerializeToString(),
      userinfo,
      artifacts.partition_assignments,
      metadata,
      compression=compression,
      hashed_database=hashed_dataset,
      float_database=float_dataset)
  with tf.io.gfile.GFile(output_file_path, 'w') as f:
    f.write(serialized_index_file)
