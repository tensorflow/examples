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
"""ScannBuilder class for on-device applications."""

from google.protobuf import text_format
from scann.proto import scann_pb2
from scann.scann_ops.py import scann_builder
from scann.scann_ops.py import scann_ops_pybind


def builder(db, num_neighbors, distance_measure):
  """pybind analogue of builder() in scann_ops.py for the on-device use case."""

  def builder_lambda(db, config, training_threads, **kwargs):
    return scann_ops_pybind.create_searcher(db, config, training_threads,
                                           **kwargs)

  return OndeviceScannBuilder(
      db, num_neighbors, distance_measure).set_builder_lambda(builder_lambda)


class OndeviceScannBuilder(scann_builder.ScannBuilder):
  """ScannBuilder for on-device applications."""

  def create_config(self):
    """Creates the config."""
    config = super().create_config()
    config_proto = scann_pb2.ScannConfig()
    text_format.Parse(config, config_proto)
    # We don't support residual quantization on device so we need to disable
    # use_residual_quantization.
    if config_proto.hash.asymmetric_hash.use_residual_quantization:
      config_proto.hash.asymmetric_hash.use_residual_quantization = False
    return text_format.MessageToString(config_proto)
