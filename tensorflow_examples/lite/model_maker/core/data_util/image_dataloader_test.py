# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import numpy as np
import tensorflow as tf # TF2

from tensorflow_examples.lite.model_maker.core.data_util import image_dataloader


def _fill_image(rgb, image_size):
  r, g, b = rgb
  return np.broadcast_to(
      np.array([[[r, g, b]]], dtype=np.uint8),
      shape=(image_size, image_size, 3))


def _write_filled_jpeg_file(path, rgb, image_size):
  tf.keras.preprocessing.image.save_img(path, _fill_image(rgb, image_size),
                                        'channels_last', 'jpeg')


class ImageDataLoaderTest(tf.test.TestCase):

  def setUp(self):
    super(ImageDataLoaderTest, self).setUp()
    self.image_path = os.path.join(self.get_temp_dir(), 'random_image_dir')
    if os.path.exists(self.image_path):
      return
    os.mkdir(self.image_path)
    for class_name in ('daisy', 'tulips'):
      class_subdir = os.path.join(self.image_path, class_name)
      os.mkdir(class_subdir)
      _write_filled_jpeg_file(
          os.path.join(class_subdir, '0.jpeg'),
          [random.uniform(0, 255) for _ in range(3)], 224)

  def test_split(self):
    ds = tf.data.Dataset.from_tensor_slices([[0, 1], [1, 1], [0, 0], [1, 0]])
    data = image_dataloader.ImageClassifierDataLoader(ds, 4, 2, ['pos', 'neg'])
    train_data, test_data = data.split(0.5)

    self.assertEqual(train_data.size, 2)
    for i, elem in enumerate(train_data.dataset):
      self.assertTrue((elem.numpy() == np.array([i, 1])).all())
    self.assertEqual(train_data.num_classes, 2)
    self.assertEqual(train_data.index_to_label, ['pos', 'neg'])

    self.assertEqual(test_data.size, 2)
    for i, elem in enumerate(test_data.dataset):
      self.assertTrue((elem.numpy() == np.array([i, 0])).all())
    self.assertEqual(test_data.num_classes, 2)
    self.assertEqual(test_data.index_to_label, ['pos', 'neg'])

  def test_from_folder(self):
    data = image_dataloader.ImageClassifierDataLoader.from_folder(
        self.image_path)

    self.assertEqual(data.size, 2)
    self.assertEqual(data.num_classes, 2)
    self.assertEqual(data.index_to_label, ['daisy', 'tulips'])
    for image, label in data.dataset:
      self.assertTrue(label.numpy() == 1 or label.numpy() == 0)
      if label.numpy() == 0:
        raw_image_tensor = image_dataloader.load_image(
            os.path.join(self.image_path, 'daisy', '0.jpeg'))
      else:
        raw_image_tensor = image_dataloader.load_image(
            os.path.join(self.image_path, 'tulips', '0.jpeg'))
      self.assertTrue((image.numpy() == raw_image_tensor.numpy()).all())

  def test_load_from_tfds(self):
    train_data, validation_data, test_data = image_dataloader.load_from_tfds(
        'beans')
    self.assertIsInstance(train_data.dataset, tf.data.Dataset)
    self.assertEqual(train_data.size, 1034)
    self.assertEqual(train_data.num_classes, 3)
    self.assertEqual(train_data.index_to_label,
                     ['angular_leaf_spot', 'bean_rust', 'healthy'])

    self.assertIsInstance(validation_data.dataset, tf.data.Dataset)
    self.assertEqual(validation_data.size, 133)
    self.assertEqual(validation_data.num_classes, 3)
    self.assertEqual(validation_data.index_to_label,
                     ['angular_leaf_spot', 'bean_rust', 'healthy'])

    self.assertIsInstance(test_data.dataset, tf.data.Dataset)
    self.assertEqual(test_data.size, 128)
    self.assertEqual(test_data.num_classes, 3)
    self.assertEqual(test_data.index_to_label,
                     ['angular_leaf_spot', 'bean_rust', 'healthy'])


if __name__ == '__main__':
  assert tf.__version__.startswith('2')
  tf.test.main()
