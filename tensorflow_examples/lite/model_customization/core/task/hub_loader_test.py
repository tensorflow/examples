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

import os

from absl import flags
from absl.testing import parameterized
import tensorflow as tf # TF2

from tensorflow_examples.lite.model_customization.core.task import hub_loader


def test_srcdir():
  """Returns the path where to look for test data files."""
  if "test_srcdir" in flags.FLAGS:
    return flags.FLAGS["test_srcdir"].value
  elif "TEST_SRCDIR" in os.environ:
    return os.environ["TEST_SRCDIR"]
  else:
    raise RuntimeError("Missing TEST_SRCDIR environment.")


def get_test_data_path(file_or_dirname):
  """Return full test data path."""
  for directory, subdirs, files in tf.io.gfile.walk(test_srcdir()):
    for f in subdirs + files:
      if f.endswith(file_or_dirname):
        return os.path.join(directory, f)
  raise ValueError("No %s in test directory" % file_or_dirname)


class HubKerasLayerV1V2Test(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(
      (("v1_implicit_tags", "hub_module_v1_mini"), ("trainable", True)),
      (("v2_implicit_tags", "saved_model_v2_mini"), ("trainable", True)),
      (("v1_implicit_tags", "hub_module_v1_mini"), ("trainable", False)),
      (("v2_implicit_tags", "saved_model_v2_mini"), ("trainable", False)),
  )
  def test_load_with_defaults(self, module_name, trainable):
    inputs, expected_outputs = 10., 11.  # Test modules perform increment op.
    path = get_test_data_path(module_name)
    layer = hub_loader.HubKerasLayerV1V2(path, trainable=trainable)
    output = layer(inputs)
    self.assertEqual(output, expected_outputs)

  def test_trainable_varaible(self):
    path = get_test_data_path("hub_module_v1_mini_train")
    layer = hub_loader.HubKerasLayerV1V2(path, trainable=True)
    self.assertLen(layer.trainable_variables, 2)
    self.assertLen(layer.variables, 4)

    layer = hub_loader.HubKerasLayerV1V2(path, trainable=False)
    self.assertEmpty(layer.trainable_variables)
    self.assertLen(layer.variables, 2)


if __name__ == "__main__":
  assert tf.__version__.startswith('2')
  tf.test.main()
