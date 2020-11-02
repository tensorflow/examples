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

import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow_examples.lite.model_maker.core.data_util import dataloader


class DataLoaderTest(tf.test.TestCase):

  def test_split(self):
    ds = tf.data.Dataset.from_tensor_slices([[0, 1], [1, 1], [0, 0], [1, 0]])
    data = dataloader.DataLoader(ds, 4)
    train_data, test_data = data.split(0.5)

    self.assertEqual(train_data.size, 2)
    self.assertIsInstance(train_data, dataloader.DataLoader)
    self.assertIsInstance(test_data, dataloader.DataLoader)
    for i, elem in enumerate(train_data.dataset):
      self.assertTrue((elem.numpy() == np.array([i, 1])).all())

    self.assertEqual(test_data.size, 2)
    for i, elem in enumerate(test_data.dataset):
      self.assertTrue((elem.numpy() == np.array([i, 0])).all())

  def test_len(self):
    size = 4
    ds = tf.data.Dataset.from_tensor_slices([[0, 1], [1, 1], [0, 0], [1, 0]])
    data = dataloader.DataLoader(ds, size)
    self.assertEqual(len(data), size)


class ClassificationDataLoaderTest(tf.test.TestCase):

  def test_split(self):

    class MagicClassificationDataLoader(dataloader.ClassificationDataLoader):

      def __init__(self, dataset, size, index_to_label, value):
        super(MagicClassificationDataLoader,
              self).__init__(dataset, size, index_to_label)
        self.value = value

      def split(self, fraction):
        return self._split(fraction, self.index_to_label, self.value)

    # Some dummy inputs.
    magic_value = 42
    num_classes = 2
    index_to_label = (False, True)

    # Create data loader from sample data.
    ds = tf.data.Dataset.from_tensor_slices([[0, 1], [1, 1], [0, 0], [1, 0]])
    data = MagicClassificationDataLoader(ds, len(ds), index_to_label,
                                         magic_value)

    # Train/Test data split.
    fraction = .25
    train_data, test_data = data.split(fraction)

    # `split` should return instances of child DataLoader.
    self.assertIsInstance(train_data, MagicClassificationDataLoader)
    self.assertIsInstance(test_data, MagicClassificationDataLoader)

    # Make sure number of entries are right.
    self.assertEqual(train_data.size, fraction * len(ds))
    self.assertEqual(test_data.size, len(ds) - len(train_data.dataset))

    # Make sure attributes propagated correctly.
    self.assertEqual(train_data.num_classes, num_classes)
    self.assertEqual(test_data.index_to_label, index_to_label)
    self.assertEqual(train_data.value, magic_value)
    self.assertEqual(test_data.value, magic_value)


if __name__ == '__main__':
  tf.test.main()
