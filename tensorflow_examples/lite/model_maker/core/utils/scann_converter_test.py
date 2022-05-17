# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for scann_converter."""

import os
import shutil

from absl import flags
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from google.protobuf import text_format
from tensorflow_lite_support.scann_ondevice.cc.core import serialized_searcher_pb2
from scann.partitioning import partitioner_pb2
from scann.proto import centers_pb2
from scann.scann_ops.py import scann_ops_pybind
from tensorflow_examples.lite.model_maker.core.utils import scann_converter
from tensorflow_lite_support.scann_ondevice.cc.test.python import leveldb_testing_utils

FLAGS = flags.FLAGS

DIMENSIONS = 20
NUM_NEIGHBORS = 10


class ScannConverterTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.artifact_dir = os.path.join(FLAGS.test_tmpdir, 'serialized_searcher')
    os.mkdir(self.artifact_dir)

  def tearDown(self):
    super().tearDown()
    shutil.rmtree(self.artifact_dir)

  @parameterized.named_parameters(
      dict(
          testcase_name='tree+ah+dot_product',
          distance='dot_product',
          tree=True,
          ah=True,
          brute_force=False),
      dict(
          testcase_name='tree+ah+squared_l2',
          distance='squared_l2',
          tree=True,
          ah=True,
          brute_force=False),
      dict(
          testcase_name='tree+bf+dot_product',
          distance='dot_product',
          tree=True,
          ah=False,
          brute_force=True),
      dict(
          testcase_name='tree+bf+squared_l2',
          distance='squared_l2',
          tree=True,
          ah=False,
          brute_force=True),
      dict(
          testcase_name='ah+dot_product',
          distance='dot_product',
          tree=False,
          ah=True,
          brute_force=False),
      dict(
          testcase_name='ah+squared_l2',
          distance='squared_l2',
          tree=False,
          ah=True,
          brute_force=False),
      dict(
          testcase_name='bf+dot_product',
          distance='dot_product',
          tree=False,
          ah=False,
          brute_force=True),
      dict(
          testcase_name='bf+squared_l2',
          distance='squared_l2',
          tree=False,
          ah=False,
          brute_force=True),
  )
  def test_converts_files_properly(self, distance: str, tree: bool, ah: bool,
                                   brute_force: bool):
    dataset_size = 10000

    dataset = np.random.random(size=(dataset_size, DIMENSIONS))
    builder = scann_ops_pybind.builder(dataset, NUM_NEIGHBORS, distance)
    if tree:
      builder = builder.tree(
          num_leaves=10, num_leaves_to_search=2, training_sample_size=1000)
    if ah:
      builder = builder.score_ah(2, anisotropic_quantization_threshold=0.2)
    if brute_force:
      builder = builder.score_brute_force(quantize=False)
    searcher = builder.build()
    searcher.serialize(self.artifact_dir)

    converted_artifacts = scann_converter.convert_serialized_to_on_device(
        self.artifact_dir)

    if distance == 'dot_product':
      self.assertEqual(converted_artifacts.ondevice_config.query_distance,
                       serialized_searcher_pb2.DOT_PRODUCT)
    if distance == 'squared_l2':
      self.assertEqual(converted_artifacts.ondevice_config.query_distance,
                       serialized_searcher_pb2.SQUARED_L2_DISTANCE)

    if ah:
      self.assertIsNotNone(converted_artifacts.hashed_dataset)
      self.assertEqual(converted_artifacts.hashed_dataset.shape,
                       (dataset_size, (DIMENSIONS + 1) // 2))
      self.assertEqual(converted_artifacts.hashed_dataset.dtype, np.uint8)
      with tf.io.gfile.GFile(
          os.path.join(self.artifact_dir, 'ah_codebook.pb'), 'rb') as ah_pb:
        ah_codebook = centers_pb2.CentersForAllSubspaces.FromString(
            ah_pb.read())
        self._verify_ah_centers(
            converted_artifacts.ondevice_config.indexer.asymmetric_hashing,
            ah_codebook)

    if brute_force:
      self.assertIsNotNone(converted_artifacts.float_dataset)
      np.testing.assert_allclose(converted_artifacts.float_dataset, dataset)

    if tree:
      self.assertIsNotNone(converted_artifacts.partition_assignments)
      self.assertEqual(converted_artifacts.partition_assignments.shape,
                       (dataset_size,))
      self.assertEqual(converted_artifacts.partition_assignments.dtype,
                       np.int32)
      np.testing.assert_array_less(converted_artifacts.partition_assignments,
                                   dataset_size)
      with tf.io.gfile.GFile(
          os.path.join(self.artifact_dir, 'serialized_partitioner.pb'),
          'rb') as partition_centroid_pb:
        partition_centroids = partitioner_pb2.SerializedPartitioner.FromString(
            partition_centroid_pb.read())
        self._verify_partition_centroids(
            converted_artifacts.ondevice_config.partitioner,
            partition_centroids)

  def _verify_ah_centers(
      self, device_ah_proto: serialized_searcher_pb2.AsymmetricHashingProto,
      desktop_ah_proto: centers_pb2.CentersForAllSubspaces) -> None:
    self.assertEqual(
        len(device_ah_proto.subspace), len(desktop_ah_proto.subspace_centers))
    for device_subspace, desktop_subspace in zip(
        device_ah_proto.subspace, desktop_ah_proto.subspace_centers):
      if device_ah_proto.lookup_type == device_ah_proto.INT8_LUT16:
        self.assertLen(device_subspace.entry, 16)
      else:
        self.assertLen(device_subspace.entry, 256)
      self.assertEqual(len(device_subspace.entry), len(desktop_subspace.center))
      for device_entry, desktop_entry in zip(device_subspace.entry,
                                             desktop_subspace.center):
        self.assertEqual(
            len(device_entry.dimension), len(desktop_entry.feature_value_float))
        self.assertListEqual(
            list(device_entry.dimension),
            list(desktop_entry.feature_value_float))

  def _verify_partition_centroids(
      self, device_partition_proto: serialized_searcher_pb2.PartitionerProto,
      desktop_partition_proto: partitioner_pb2.SerializedPartitioner):
    self.assertEqual(
        len(device_partition_proto.leaf),
        len(desktop_partition_proto.kmeans.kmeans_tree.root.centers))
    for device_centroid, deesktop_centroid in zip(
        device_partition_proto.leaf,
        desktop_partition_proto.kmeans.kmeans_tree.root.centers):
      self.assertEqual(
          len(device_centroid.dimension), len(deesktop_centroid.dimension))
      self.assertListEqual(
          list(device_centroid.dimension), list(deesktop_centroid.dimension))

  @parameterized.named_parameters(
      dict(testcase_name='hashed_compressed', hashed=True, compressed=True),
      dict(testcase_name='hashed_uncompressed', hashed=True, compressed=False),
      dict(testcase_name='float_compressed', hashed=False, compressed=True),
      dict(testcase_name='float_uncompressed', hashed=False, compressed=False),
  )
  def test_generates_index_leveldb_file(self, hashed: bool, compressed: bool):
    dataset_size = 10

    config = serialized_searcher_pb2.ScannOnDeviceConfig()
    text_format.Parse("""
    partitioner: {
      leaf: { dimension: 1 dimension: 1 dimension: 1 }
      leaf: { dimension: 2 dimension: 2 dimension: 2 }
      leaf: { dimension: 3 dimension: 3 dimension: 3 }
    }
    """, config)
    if hashed:
      hashed_dataset = np.array(
          [i // DIMENSIONS for i in range(dataset_size * DIMENSIONS)],
          dtype=np.uint8).reshape((dataset_size, DIMENSIONS))
      float_dataset = None
    else:
      float_dataset = np.array(
          [i // DIMENSIONS for i in range(dataset_size * DIMENSIONS)],
          dtype=np.float32).reshape((dataset_size, DIMENSIONS))
      hashed_dataset = None
    partition_assignments = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    metadata = ['0', '1', '2', '3', '你好', b'\x00\x22\x33', '6', '7', '8', '9']

    output_file_path = os.path.join(self.artifact_dir, 'index_file')
    scann_converter.convert_artifacts_to_leveldb(
        output_file_path=output_file_path,
        metadata=metadata,
        userinfo=b'userinfo\x00\x11\x22\x34',
        artifacts=scann_converter.OnDeviceArtifacts(
            ondevice_config=config,
            hashed_dataset=hashed_dataset,
            float_dataset=float_dataset,
            partition_assignments=partition_assignments),
        compression=compressed)

    db_contents = dict(
        leveldb_testing_utils.leveldb_table_to_pair_list(
            output_file_path, compressed))

    partition_2 = np.frombuffer(
        db_contents[b'E_2'],
        dtype=np.uint8 if hashed else np.float32).reshape((3, DIMENSIONS))

    if hashed:
      expected_dataset = hashed_dataset
    else:
      expected_dataset = float_dataset
    np.testing.assert_array_equal(partition_2[0], expected_dataset[2])
    np.testing.assert_array_equal(partition_2[1], expected_dataset[5])
    np.testing.assert_array_equal(partition_2[2], expected_dataset[8])

    config = serialized_searcher_pb2.ScannOnDeviceConfig()

    self.assertEqual(db_contents[b'USER_INFO'], b'userinfo\x00\x11\x22\x34')

    # Original embedding index after reordering by partitions
    # [0, 3, 6, 9, 1, 4, 7, 2, 5, 8]
    self.assertEqual(db_contents[b'M_1'], b'3')
    self.assertEqual(db_contents[b'M_5'], '你好'.encode('utf-8'))
    self.assertEqual(db_contents[b'M_8'], b'\x00\x22\x33')


if __name__ == '__main__':
  tf.test.main()
