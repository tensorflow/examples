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

import filecmp
import os

import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow_examples.lite.model_maker.core import compat
from tensorflow_examples.lite.model_maker.core import test_util
from tensorflow_examples.lite.model_maker.core.data_util import image_dataloader
from tensorflow_examples.lite.model_maker.core.export_format import ExportFormat
from tensorflow_examples.lite.model_maker.core.task import configs
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

  @test_util.test_in_tf_2
  def test_mobilenetv2_model(self):
    model = image_classifier.create(
        self.train_data,
        model_spec.mobilenet_v2_spec,
        epochs=2,
        batch_size=4,
        shuffle=True)
    self._test_accuracy(model)
    self._test_predict_top_k(model)
    self._test_export_to_tflite(model)
    self._test_export_to_tflite_quantized(model, self.train_data)
    self._test_export_to_tflite_with_metadata(model)
    self._test_export_to_saved_model(model)
    self._test_export_labels(model)

  @test_util.test_in_tf_1
  def test_mobilenetv2_model_create_v1_incompatible(self):
    with self.assertRaisesRegex(ValueError, 'Incompatible versions'):
      _ = image_classifier.create(self.train_data, model_spec.mobilenet_v2_spec)

  @test_util.test_in_tf_1and2
  def test_efficientnetlite0_model_with_model_maker_retraining_lib(self):
    model = image_classifier.create(
        self.train_data,
        model_spec.efficientnet_lite0_spec,
        epochs=2,
        batch_size=4,
        shuffle=True,
        use_hub_library=False)
    self._test_accuracy(model)
    self._test_export_to_tflite(model)

  @test_util.test_in_tf_1and2
  def test_efficientnetlite0_model(self):
    model = image_classifier.create(
        self.train_data,
        model_spec.efficientnet_lite0_spec,
        epochs=2,
        batch_size=4,
        shuffle=True)
    self._test_accuracy(model)
    self._test_export_to_tflite(model)
    self._test_export_to_tflite_quantized(model, self.train_data)
    self._test_export_to_tflite_with_metadata(
        model, expected_json_file='efficientnet_lite0_metadata.json')

  @test_util.test_in_tf_1and2
  def test_efficientnetlite0_model_without_training(self):
    model = image_classifier.create(
        self.train_data, model_spec.efficientnet_lite0_spec, do_train=False)
    self._test_accuracy(model, threshold=0.0)
    self._test_export_to_tflite(model, threshold=0.0)

  @test_util.test_in_tf_2
  def test_resnet_50_model(self):
    model = image_classifier.create(
        self.train_data,
        model_spec.resnet_50_spec,
        epochs=2,
        batch_size=4,
        shuffle=True)
    self._test_accuracy(model)
    self._test_export_to_tflite(model)
    self._test_export_to_tflite_quantized(model, self.train_data)
    self._test_export_to_tflite_with_metadata(model)

  def _test_predict_top_k(self, model, threshold=0.7):
    topk = model.predict_top_k(self.test_data, batch_size=4)
    for i, (_, label) in enumerate(self.test_data.dataset):
      predict_label, predict_prob = topk[i][0][0], topk[i][0][1]
      self.assertEqual(model.index_to_label[label], predict_label)
      self.assertGreater(predict_prob, threshold)

  def _test_accuracy(self, model, threshold=0.8):
    _, accuracy = model.evaluate(self.test_data)
    self.assertGreater(accuracy, threshold)

  def _load_labels(self, filename):
    with tf.io.gfile.GFile(filename, 'r') as f:
      return [label.strip() for label in f]

  def _test_export_labels(self, model):
    labels_output_file = os.path.join(self.get_temp_dir(), 'labels.txt')
    model.export(self.get_temp_dir(), export_format=ExportFormat.LABEL)
    self._check_label_file(labels_output_file)

  def _test_export_to_tflite(self, model, threshold=1.0):
    tflite_output_file = os.path.join(self.get_temp_dir(), 'model.tflite')

    model.export(self.get_temp_dir(), export_format=ExportFormat.TFLITE)

    result = model.evaluate_tflite(tflite_output_file, self.test_data)
    self.assertGreaterEqual(result['accuracy'], threshold)

    random_input = np.random.uniform(
        size=[1] + model.model_spec.input_image_shape + [3]).astype(np.float32)
    self.assertTrue(
        test_util.is_same_output(tflite_output_file, model.model, random_input,
                                 model.model_spec))

  def _test_export_to_tflite_quantized(self, model, representative_data):
    # Just test whether quantization will crash, can't guarantee the result.
    tflile_filename = 'model_quantized.tflite'
    tflite_output_file = os.path.join(self.get_temp_dir(), tflile_filename)
    config = configs.QuantizationConfig.create_full_integer_quantization(
        representative_data, is_integer_only=True)
    model.export(
        self.get_temp_dir(),
        tflile_filename,
        quantization_config=config,
        export_format=ExportFormat.TFLITE)
    self.assertTrue(os.path.isfile(tflite_output_file))
    self.assertGreater(os.path.getsize(tflite_output_file), 0)

  def _check_label_file(self, labels_output_file):
    labels = self._load_labels(labels_output_file)
    self.assertEqual(labels, ['cyan', 'magenta', 'yellow'])

  def _test_export_to_tflite_with_metadata(self,
                                           model,
                                           expected_json_file=None):
    model_name = 'model_with_metadata'
    tflite_output_file = os.path.join(self.get_temp_dir(),
                                      '%s.tflite' % model_name)
    json_output_file = os.path.join(self.get_temp_dir(), '%s.json' % model_name)
    labels_output_file = os.path.join(self.get_temp_dir(), 'labels.txt')

    model.export(
        self.get_temp_dir(),
        '%s.tflite' % model_name,
        with_metadata=True,
        export_metadata_json_file=True)

    self.assertTrue(os.path.isfile(tflite_output_file))
    self.assertGreater(os.path.getsize(tflite_output_file), 0)

    self.assertFalse(os.path.isfile(labels_output_file))

    self.assertTrue(os.path.isfile(json_output_file))
    self.assertGreater(os.path.getsize(json_output_file), 0)

    if expected_json_file is not None:
      expected_json_file = test_util.get_test_data_path(expected_json_file)
      self.assertTrue(filecmp.cmp(json_output_file, expected_json_file))

  def _test_export_to_saved_model(self, model):
    save_model_output_path = os.path.join(self.get_temp_dir(), 'saved_model')
    model.export(self.get_temp_dir(), export_format=ExportFormat.SAVED_MODEL)

    self.assertTrue(os.path.isdir(save_model_output_path))
    self.assertNotEqual(len(os.listdir(save_model_output_path)), 0)


if __name__ == '__main__':
  compat.setup_tf_behavior(tf_version=2)
  tf.test.main()
