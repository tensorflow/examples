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
"""Tests for ondevice_scann_builder."""

import numpy as np
import tensorflow as tf
from google.protobuf import text_format
from scann.proto import scann_pb2
from tensorflow_examples.lite.model_maker.core.utils import ondevice_scann_builder


class OndeviceScannBuilderTest(tf.test.TestCase):

  def test_create_from_config_bruteforce(self):
    dataset = np.random.random(size=(1000, 1024))
    builder = ondevice_scann_builder.builder(
        dataset, num_neighbors=10, distance_measure="dot_product")
    builder.score_brute_force()
    config = builder.create_config()
    config_proto = scann_pb2.ScannConfig()
    text_format.Parse(config, config_proto)
    expected_config_proto = """num_neighbors: 10
        distance_measure {
          distance_measure: "DotProductDistance"
        }
        brute_force {
          fixed_point {
            enabled: false
          }
        }"""
    self.assertProtoEquals(expected_config_proto, config_proto)

  def test_create_from_config_ah(self):
    dataset = np.random.random(size=(1000, 1024))
    builder = ondevice_scann_builder.builder(
        dataset, num_neighbors=5, distance_measure="dot_product")
    builder.tree(num_leaves=10, num_leaves_to_search=2)
    builder.score_ah(
        dimensions_per_block=1,
        anisotropic_quantization_threshold=0.2,
        hash_type="lut256")
    config = builder.create_config()
    config_proto = scann_pb2.ScannConfig()
    text_format.Parse(config, config_proto)
    # Changes use_residual_quantization = False.
    expected_config_proto = """num_neighbors: 5
        distance_measure {
          distance_measure: "DotProductDistance"
        }
        partitioning {
          num_children: 10
          max_clustering_iterations: 12
          min_cluster_size: 50.0
          partitioning_distance {
            distance_measure: "SquaredL2Distance"
          }
          query_spilling {
            spilling_type: FIXED_NUMBER_OF_CENTERS
            max_spill_centers: 2
          }
          partitioning_type: GENERIC
          query_tokenization_distance_override {
            distance_measure: "DotProductDistance"
          }
          query_tokenization_type: FLOAT
          expected_sample_size: 100000
          single_machine_center_initialization: RANDOM_INITIALIZATION
        }
        hash {
          asymmetric_hash {
            projection {
              projection_type: CHUNK
              num_blocks: 1024
              num_dims_per_block: 1
              input_dim: 1024
            }
            num_clusters_per_block: 256
            max_clustering_iterations: 10
            quantization_distance {
              distance_measure: "SquaredL2Distance"
            }
            min_cluster_size: 100.0
            lookup_type: INT8
            use_residual_quantization: false
            fixed_point_lut_conversion_options {
              float_to_int_conversion_method: ROUND
            }
            noise_shaping_threshold: 0.2
            expected_sample_size: 100000
            use_global_topn: false
          }
        }"""
    self.assertProtoEquals(expected_config_proto, config_proto)


if __name__ == "__main__":
  tf.test.main()
