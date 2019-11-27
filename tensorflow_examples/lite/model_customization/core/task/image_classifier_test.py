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
import tensorflow as tf # TF2
from tensorflow_examples.lite.model_customization.core.data_util import image_dataloader
import tensorflow_examples.lite.model_customization.core.model_export_format as mef
from tensorflow_examples.lite.model_customization.core.task import image_classifier
from tensorflow_examples.lite.model_customization.core.task import model_spec


def _fill_image(rgb, image_size):
  r, g, b = rgb
  return np.broadcast_to(
      np.array([[[r, g, b]]], dtype=np.uint8),
      shape=(image_size, image_size, 3))


class ImageClassifierTest(tf.test.TestCase):
  IMAGE_SIZE = 24
  IMAGES_PER_CLASS = 20
  CMY_NAMES_AND_RGB_VALUES = (('cyan', (0, 255, 255)),
                              ('magenta', (255, 0, 255)), ('yellow', (255, 255,
                                                                      0)))

  def _gen(self):
    for i, (_, rgb) in enumerate(self.CMY_NAMES_AND_RGB_VALUES):
      for _ in range(self.IMAGES_PER_CLASS):
        yield (_fill_image(rgb, self.IMAGE_SIZE), i)

  def _gen_cmy_data(self):
    ds = tf.data.Dataset.from_generator(
        self._gen, (tf.uint8, tf.int64), (tf.TensorShape(
            [self.IMAGE_SIZE, self.IMAGE_SIZE, 3]), tf.TensorShape([])))
    data = image_dataloader.ImageClassifierDataLoader(
        ds, self.IMAGES_PER_CLASS * 3, 3, ['cyan', 'magenta', 'yellow'])
    return data

  def setUp(self):
    super(ImageClassifierTest, self).setUp()
    self.data = self._gen_cmy_data()

  def test_mobilenetv2_model(self):
    model = image_classifier.create(
        self.data,
        mef.ModelExportFormat.TFLITE,
        model_spec.mobilenet_v2_spec,
        epochs=2,
        batch_size=4,
        shuffle=True)
    self._test_accuracy(model)
    self._test_export_to_tflite(model)
    self._test_predict_topk(model)

  def test_efficientnetb0_model(self):
    model = image_classifier.create(
        self.data,
        mef.ModelExportFormat.TFLITE,
        model_spec.efficientnet_b0_spec,
        epochs=5,
        batch_size=4,
        shuffle=True)
    self._test_accuracy(model)
    self._test_export_to_tflite(model)

  def _test_predict_topk(self, model):
    topk = model.predict_topk(batch_size=4)
    for i, (_, label) in enumerate(model.test_data.dataset):
      predict_label, predict_prob = topk[i][0][0], topk[i][0][1]
      self.assertEqual(model.data.index_to_label[label], predict_label)
      self.assertGreater(predict_prob, 0.7)

  def _test_accuracy(self, model):
    _, accuracy = model.evaluate()
    self.assertEqual(accuracy, 1.0)

  def _load_labels(self, filename):
    with tf.io.gfile.GFile(filename, 'r') as f:
      return [label.strip('\n') for label in f]

  def _load_lite_model(self, filename):
    self.assertTrue(os.path.isfile(filename))
    with tf.io.gfile.GFile(filename, 'rb') as f:
      model_content = f.read()
    interpreter = tf.lite.Interpreter(model_content=model_content)

    def lite_model(images):
      interpreter.allocate_tensors()
      input_index = interpreter.get_input_details()[0]['index']
      interpreter.set_tensor(input_index, images)
      interpreter.invoke()
      output_index = interpreter.get_output_details()[0]['index']
      return interpreter.get_tensor(output_index)

    return lite_model

  def _test_export_to_tflite(self, model):
    tflite_output_file = os.path.join(self.get_temp_dir(), 'model.tflite')
    labels_output_file = os.path.join(self.get_temp_dir(), 'label')
    model.export(tflite_output_file, labels_output_file)
    labels = self._load_labels(labels_output_file)
    self.assertEqual(labels, ['cyan', 'magenta', 'yellow'])
    lite_model = self._load_lite_model(tflite_output_file)
    for i, (class_name, rgb) in enumerate(self.CMY_NAMES_AND_RGB_VALUES):
      input_batch = tf.constant(_fill_image(rgb, self.IMAGE_SIZE))
      input_batch = model.preprocess_image(input_batch, i)[0]
      input_batch = tf.expand_dims(input_batch, 0).numpy()
      output_batch = lite_model(input_batch)
      prediction = labels[np.argmax(output_batch[0])]
      self.assertEqual(class_name, prediction)


if __name__ == '__main__':
  tf.compat.v1.enable_v2_behavior()
  tf.test.main()
