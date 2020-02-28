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
from tensorflow_examples.lite.model_maker.core import compat
from tensorflow_examples.lite.model_maker.core import model_export_format as mef
from tensorflow_examples.lite.model_maker.core.data_util import image_dataloader
from tensorflow_examples.lite.model_maker.core.task import image_classifier
from tensorflow_examples.lite.model_maker.core.task import model_spec


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
    all_data = self._gen_cmy_data()
    # Splits data, 90% data for training, 10% for testing
    self.train_data, self.test_data = all_data.split(0.9)

  @compat.test_in_tf_2
  def test_mobilenetv2_model(self):
    model = image_classifier.create(
        self.train_data,
        mef.ModelExportFormat.TFLITE,
        model_spec.mobilenet_v2_spec,
        epochs=2,
        batch_size=4,
        shuffle=True)
    self._test_accuracy(model)
    self._test_export_to_tflite(model)
    self._test_predict_top_k(model)
    self._test_export_to_tflite_quantized(model, self.train_data)

  @compat.test_in_tf_1
  def test_mobilenetv2_model_create_v1_incompatible(self):
    with self.assertRaisesRegex(ValueError, 'Incompatible versions'):
      _ = image_classifier.create(self.train_data, mef.ModelExportFormat.TFLITE,
                                  model_spec.mobilenet_v2_spec)

  @compat.test_in_tf_1and2
  def test_efficientnetb0_model(self):
    model = image_classifier.create(
        self.train_data,
        mef.ModelExportFormat.TFLITE,
        model_spec.efficientnet_b0_spec,
        epochs=2,
        batch_size=4,
        shuffle=True)
    self._test_accuracy(model)
    self._test_export_to_tflite(model)

  @compat.test_in_tf_2
  def test_resnet_50_model(self):
    model = image_classifier.create(
        self.train_data,
        mef.ModelExportFormat.TFLITE,
        model_spec.resnet_50_spec,
        epochs=2,
        batch_size=4,
        shuffle=True)
    self._test_accuracy(model)
    self._test_export_to_tflite(model)

  def _test_predict_top_k(self, model, threshold=0.7):
    topk = model.predict_top_k(self.test_data, batch_size=4)
    for i, (_, label) in enumerate(self.test_data.dataset):
      predict_label, predict_prob = topk[i][0][0], topk[i][0][1]
      self.assertEqual(model.index_to_label[label], predict_label)
      self.assertGreater(predict_prob, threshold)

  def _test_accuracy(self, model, threashold=0.8):
    _, accuracy = model.evaluate(self.test_data)
    self.assertGreater(accuracy, threashold)

  def _load_labels(self, filename):
    with tf.io.gfile.GFile(filename, 'r') as f:
      return [label.strip() for label in f]

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

    if compat.get_tf_behavior() == 1:
      image_placeholder = tf.compat.v1.placeholder(
          tf.uint8, [1, self.IMAGE_SIZE, self.IMAGE_SIZE, 3])
      label_placeholder = tf.compat.v1.placeholder(tf.int32, [1])
      image_tensor, _ = model.preprocess(image_placeholder, label_placeholder)
      with tf.compat.v1.Session() as sess:
        for i, (class_name, rgb) in enumerate(self.CMY_NAMES_AND_RGB_VALUES):
          input_image = np.expand_dims(_fill_image(rgb, self.IMAGE_SIZE), 0)
          image = sess.run(
              image_tensor,
              feed_dict={
                  image_placeholder: input_image,
                  label_placeholder: [i]
              })
          output_batch = lite_model(image)
          prediction = labels[np.argmax(output_batch[0])]
          self.assertEqual(class_name, prediction)
    else:
      for i, (class_name, rgb) in enumerate(self.CMY_NAMES_AND_RGB_VALUES):
        input_batch = np.expand_dims(_fill_image(rgb, self.IMAGE_SIZE), 0)
        image, _ = model.preprocess(input_batch, i)
        image = image.numpy()
        output_batch = lite_model(image)
        prediction = labels[np.argmax(output_batch[0])]
        self.assertEqual(class_name, prediction)

  def _test_export_to_tflite_quantized(self, model, representative_data):
    # Just test whether quantization will crash, can't guarantee the result.
    tflite_output_file = os.path.join(self.get_temp_dir(), 'model.tflite')
    labels_output_file = os.path.join(self.get_temp_dir(), 'label')
    model.export(
        tflite_output_file,
        labels_output_file,
        quantized=True,
        representative_data=representative_data)
    self.assertTrue(
        os.path.isfile(tflite_output_file) and
        os.path.getsize(tflite_output_file) > 0)
    labels = self._load_labels(labels_output_file)
    self.assertEqual(labels, ['cyan', 'magenta', 'yellow'])


if __name__ == '__main__':
  compat.setup_tf_behavior(tf_version=2)
  tf.test.main()
