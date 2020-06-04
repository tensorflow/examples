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

  def test_export_tflite(self):
    input_dim = 4
    model = test_util.build_model(input_shape=[input_dim], num_classes=2)
    tflite_file = os.path.join(self.get_temp_dir(), 'model.tflite')
    model_util.export_tflite(model, tflite_file)
    self._test_tflite(model, tflite_file, input_dim)

  def test_export_tflite_quantized(self):
    input_dim = 4
    num_classes = 2
    max_input_value = 5
    model = test_util.build_model([input_dim], num_classes)
    tflite_file = os.path.join(self.get_temp_dir(), 'model_quantized.tflite')

    for config in _get_quantization_config_list(input_dim, num_classes,
                                                max_input_value):
      model_util.export_tflite(model, tflite_file, config, _mock_gen_dataset)
      self._test_tflite(
          model, tflite_file, input_dim, max_input_value, atol=1e-01)

  def _test_tflite(self,
                   keras_model,
                   tflite_model_file,
                   input_dim,
                   max_input_value=1000,
                   atol=1e-04):
    with tf.io.gfile.GFile(tflite_model_file, 'rb') as f:
      tflite_model = f.read()

    np.random.seed(0)
    random_input = np.random.uniform(
        low=0, high=max_input_value, size=(1, input_dim)).astype(np.float32)

    # Gets output from keras model.
    keras_output = keras_model.predict(random_input)

    # Gets output from tflite model.
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    if input_details['dtype'] != np.float32:
      # Quantize the input
      scale, zero_point = input_details['quantization']
      random_input = random_input / scale + zero_point
      random_input = random_input.astype(input_details['dtype'])
    interpreter.set_tensor(input_details['index'], random_input)
    interpreter.invoke()
    output_details = interpreter.get_output_details()[0]
    lite_output = interpreter.get_tensor(output_details['index'])
    if output_details['dtype'] != np.float32:
      # Dequantize the output
      scale, zero_point = output_details['quantization']
      lite_output = lite_output.astype(np.float32)
      lite_output = (lite_output - zero_point) * scale
    self.assertTrue(np.allclose(lite_output, keras_output, atol=atol))


if __name__ == '__main__':
  tf.test.main()
