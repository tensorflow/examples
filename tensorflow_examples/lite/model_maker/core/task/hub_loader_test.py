# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import tensorflow as tf

from tensorflow_examples.lite.model_maker.core import test_util
from tensorflow_examples.lite.model_maker.core.task import hub_loader


class HubKerasLayerV1V2Test(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(
      ("hub_module_v1_mini", True),
      ("saved_model_v2_mini", True),
      ("hub_module_v1_mini", False),
      ("saved_model_v2_mini", False),
  )
  def test_load_with_defaults(self, module_name, trainable):
    inputs, expected_outputs = 10., 11.  # Test modules perform increment op.
    path = test_util.get_test_data_path(module_name)
    layer = hub_loader.HubKerasLayerV1V2(path, trainable=trainable)
    output = layer(inputs)
    self.assertEqual(output, expected_outputs)

  def test_trainable_varaible(self):
    path = test_util.get_test_data_path("hub_module_v1_mini_train")
    layer = hub_loader.HubKerasLayerV1V2(path, trainable=True)
    # Checks trainable variables.
    self.assertLen(layer.trainable_variables, 2)
    self.assertEqual(layer.trainable_variables[0].name, "a:0")
    self.assertEqual(layer.trainable_variables[1].name, "b:0")
    self.assertEqual(layer.variables, layer.trainable_variables)
    # Checks non-trainable variables.
    self.assertEmpty(layer.non_trainable_variables)

    layer = hub_loader.HubKerasLayerV1V2(path, trainable=False)
    # Checks trainable variables.
    self.assertEmpty(layer.trainable_variables)
    # Checks non-trainable variables.
    self.assertLen(layer.non_trainable_variables, 2)
    self.assertEqual(layer.non_trainable_variables[0].name, "a:0")
    self.assertEqual(layer.non_trainable_variables[1].name, "b:0")
    self.assertEqual(layer.variables, layer.non_trainable_variables)


if __name__ == "__main__":
  tf.test.main()
