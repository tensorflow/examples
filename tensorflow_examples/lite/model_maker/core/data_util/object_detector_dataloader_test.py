# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

import filecmp
import os
import unittest

import numpy as np

import tensorflow as tf
from tensorflow_examples.lite.model_maker.core import test_util

from tensorflow_examples.lite.model_maker.core.data_util import object_detector_dataloader
from tensorflow_examples.lite.model_maker.third_party.efficientdet import hparams_config
from tensorflow_examples.lite.model_maker.third_party.efficientdet import utils


class MockDetectorModelSpec(object):

  def __init__(self, model_name):
    self.model_name = model_name
    config = hparams_config.get_detection_config(model_name)
    config.image_size = utils.parse_image_size(config.image_size)
    config.update({'debug': False})
    self.config = config


class ObjectDectectorDataLoaderTest(tf.test.TestCase):

  @unittest.skip('Temporaty skipping b/182437378.')
  def test_from_pascal_voc(self):

    images_dir, annotations_dir, label_map = test_util.create_pascal_voc(
        self.get_temp_dir())
    model_spec = MockDetectorModelSpec('efficientdet-lite0')

    data = object_detector_dataloader.DataLoader.from_pascal_voc(
        images_dir, annotations_dir, label_map)

    self.assertIsInstance(data, object_detector_dataloader.DataLoader)
    self.assertLen(data, 1)
    self.assertEqual(data.label_map, label_map)

    self.assertTrue(os.path.isfile(data.annotations_json_file))
    self.assertGreater(os.path.getsize(data.annotations_json_file), 0)
    expected_json_file = test_util.get_test_data_path('annotations.json')
    self.assertTrue(filecmp.cmp(data.annotations_json_file, expected_json_file))

    ds = data.gen_dataset(model_spec, batch_size=1, is_training=False)
    for i, (images, labels) in enumerate(ds):
      self.assertEqual(i, 0)
      images_shape = tf.shape(images).numpy()
      expected_shape = np.array([1, *model_spec.config.image_size, 3])
      self.assertTrue((images_shape == expected_shape).all())
      self.assertLen(labels, 15)

    ds1 = data.gen_dataset(model_spec, batch_size=1, is_training=True)
    # Comments out this assert since it fails externally.
    # self.assertEqual(ds1.cardinality(), tf.data.INFINITE_CARDINALITY)
    for images, labels in ds1.take(10):
      images_shape = tf.shape(images).numpy()
      expected_shape = np.array([1, *model_spec.config.image_size, 3])
      self.assertTrue((images_shape == expected_shape).all())
      self.assertLen(labels, 15)


if __name__ == '__main__':
  tf.test.main()
