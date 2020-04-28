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
import tensorflow.compat.v2 as tf
from tensorflow_examples.lite.model_maker.core import test_util
from tensorflow_examples.lite.model_maker.core.task import custom_model


class MockCustomModel(custom_model.CustomModel):

  def train(self, train_data, validation_data=None, **kwargs):
    pass

  def export(self, **kwargs):
    pass

  def evaluate(self, data, **kwargs):
    pass


class CustomModelTest(tf.test.TestCase):

  def setUp(self):
    super(CustomModelTest, self).setUp()
    self.model = MockCustomModel(
        model_spec=None,
        shuffle=False)

  def test_gen_dataset(self):
    input_dim = 8
    data = test_util.get_dataloader(
        data_size=2, input_shape=[input_dim], num_classes=2)

    ds = self.model._gen_dataset(data, batch_size=1, is_training=False)
    expected = list(data.dataset.as_numpy_iterator())
    for i, (feature, label) in enumerate(ds):
      expected_feature = [expected[i][0]]
      expected_label = [expected[i][1]]
      self.assertTrue((feature.numpy() == expected_feature).any())
      self.assertEqual(label.numpy(), expected_label)

  def test_export_tflite(self):
    input_dim = 4
    self.model.model = test_util.build_model(
        input_shape=[input_dim], num_classes=2)
    tflite_file = os.path.join(self.get_temp_dir(), 'model.tflite')
    self.model._export_tflite(tflite_file)
    self._test_tflite(self.model.model, tflite_file, input_dim)

  def test_export_saved_model(self):
    self.model.model = test_util.build_model(input_shape=[4], num_classes=2)
    saved_model_filepath = os.path.join(self.get_temp_dir(), 'saved_model/')
    self.model._export_saved_model(saved_model_filepath)
    self.assertTrue(os.path.isdir(saved_model_filepath))
    self.assertNotEqual(len(os.listdir(saved_model_filepath)), 0)

  def test_export_tflite_quantized(self):
    input_dim = 4
    num_classes = 2
    max_input_value = 5
    self.model.model = test_util.build_model([input_dim], num_classes)
    tflite_file = os.path.join(self.get_temp_dir(), 'model_quantized.tflite')
    self.model._export_tflite(
        tflite_file,
        quantized=True,
        quantization_steps=1,
        representative_data=test_util.get_dataloader(
            data_size=1,
            input_shape=[input_dim],
            num_classes=num_classes,
            max_input_value=max_input_value))
    self._test_tflite(
        self.model.model, tflite_file, input_dim, max_input_value, atol=1e-01)

  def _test_tflite(self,
                   keras_model,
                   tflite_model_file,
                   input_dim,
                   max_input_value=1000,
                   atol=1e-04):
    with tf.io.gfile.GFile(tflite_model_file, 'rb') as f:
      tflite_model = f.read()

    random_input = tf.random.uniform(
        shape=(1, input_dim),
        minval=0,
        maxval=max_input_value,
        dtype=tf.float32)

    # Gets output from keras model.
    keras_output = keras_model.predict(random_input)

    # Gets output from tflite model.
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'],
                           random_input)
    interpreter.invoke()
    lite_output = interpreter.get_tensor(
        interpreter.get_output_details()[0]['index'])

    self.assertTrue(np.allclose(lite_output, keras_output, atol=atol))


if __name__ == '__main__':
  tf.test.main()
