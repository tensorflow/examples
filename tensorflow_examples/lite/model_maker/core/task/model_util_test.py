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

import numpy as np
import tensorflow as tf
from tensorflow_examples.lite.model_maker.core import test_util
from tensorflow_examples.lite.model_maker.core.task import configs
from tensorflow_examples.lite.model_maker.core.task import model_util


def _get_quantization_config_list(input_dim, num_classes, max_input_value):
  # Configuration for dynamic range quantization.
  config1 = configs.QuantizationConfig.create_dynamic_range_quantization()

  representative_data = test_util.get_dataloader(
      data_size=1,
      input_shape=[input_dim],
      num_classes=num_classes,
      max_input_value=max_input_value)
  # Configuration for full integer quantization with float fallback.
  config2 = configs.QuantizationConfig.create_full_integer_quantization(
      representative_data=representative_data, quantization_steps=1)
  # Configuration for full integer quantization with integer only.
  config3 = configs.QuantizationConfig.create_full_integer_quantization(
      representative_data=representative_data,
      quantization_steps=1,
      is_integer_only=True)

  # Configuration for full integer quantization with float fallback.
  config4 = configs.QuantizationConfig.create_float16_quantization()
  return [config1, config2, config3, config4]


def _mock_gen_dataset(data, batch_size=1, is_training=False):  # pylint: disable=unused-argument
  ds = data.dataset
  ds = ds.batch(batch_size)
  return ds


class ModelUtilTest(tf.test.TestCase):

  @test_util.test_in_tf_1and2
  def test_export_tflite(self):
    input_dim = 4
    model = test_util.build_model(input_shape=[input_dim], num_classes=2)
    tflite_file = os.path.join(self.get_temp_dir(), 'model.tflite')
    model_util.export_tflite(model, tflite_file)
    self._test_tflite(model, tflite_file, input_dim)

  @test_util.test_in_tf_1and2
  def test_export_tflite_quantized(self):
    input_dim = 4000
    num_classes = 2
    max_input_value = 5
    model = test_util.build_model([input_dim], num_classes)
    tflite_file = os.path.join(self.get_temp_dir(), 'model_quantized.tflite')

    quant_configs = _get_quantization_config_list(input_dim, num_classes,
                                                  max_input_value)
    model_sizes = [9088, 9536, 9600, 17280]
    for config, model_size in zip(quant_configs, model_sizes):
      model_util.export_tflite(model, tflite_file, config, _mock_gen_dataset)
      self._test_tflite(
          model, tflite_file, input_dim, max_input_value, atol=1e-01)
      self.assertNear(os.path.getsize(tflite_file), model_size, 300)

  def _test_tflite(self,
                   keras_model,
                   tflite_model_file,
                   input_dim,
                   max_input_value=1000,
                   atol=1e-04):
    np.random.seed(0)
    random_input = np.random.uniform(
        low=0, high=max_input_value, size=(1, input_dim)).astype(np.float32)

    self.assertTrue(
        test_util.is_same_output(
            tflite_model_file, keras_model, random_input, atol=atol))

  @test_util.test_in_tf_1and2
  def test_export_tfjs(self):
    input_dim = 4000
    num_classes = 2
    model = test_util.build_model([input_dim], num_classes)

    saved_model_dir = os.path.join(self.get_temp_dir(), 'saved_model_for_js')
    model.save(saved_model_dir)

    output_dir = os.path.join(self.get_temp_dir(), 'tfjs')

    model_util.export_tfjs(saved_model_dir, output_dir)

    if model_util.HAS_TFJS:
      self.assertTrue(os.path.exists(output_dir))
      expected_model_json = os.path.join(output_dir, 'model.json')
      self.assertTrue(os.path.exists(expected_model_json))


if __name__ == '__main__':
  tf.test.main()
