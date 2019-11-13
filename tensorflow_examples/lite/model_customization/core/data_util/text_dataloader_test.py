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
import numpy as np
import tensorflow as tf # TF2

from tensorflow_examples.lite.model_customization.core.data_util import text_dataloader


class TextDataLoaderTest(tf.test.TestCase):

  def setUp(self):
    super(TextDataLoaderTest, self).setUp()
    self.text_path = os.path.join(self.get_temp_dir(), 'random_text_dir')
    if os.path.exists(self.text_path):
      return
    os.mkdir(self.text_path)
    text_dir = {'neg': 'so bad', 'pos': 'really good'}
    for class_name in ('neg', 'pos'):
      class_subdir = os.path.join(self.text_path, class_name)
      os.mkdir(class_subdir)
      with open(os.path.join(class_subdir, '0.txt'), 'w') as f:
        f.write(text_dir[class_name])

  def test_split(self):
    ds = tf.data.Dataset.from_tensor_slices([[0, 1], [1, 1], [0, 0], [1, 0]])
    data = text_dataloader.TextClassifierDataLoader(ds, 4, 2, ['pos', 'neg'])
    train_data, test_data = data.split(0.5, shuffle=False)

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
    data = text_dataloader.TextClassifierDataLoader.from_folder(self.text_path)

    self.assertEqual(data.size, 2)
    self.assertEqual(data.num_classes, 2)
    self.assertEqual(data.index_to_label, ['neg', 'pos'])
    for text, label in data.dataset:
      self.assertTrue(label.numpy() == 1 or label.numpy() == 0)
      if label.numpy() == 0:
        raw_text_tensor = text_dataloader.load_text(
            os.path.join(self.text_path, 'neg', '0.txt'))
      else:
        raw_text_tensor = text_dataloader.load_text(
            os.path.join(self.text_path, 'pos', '0.txt'))
      self.assertEqual(text.numpy(), raw_text_tensor.numpy())


if __name__ == '__main__':
  tf.compat.v1.enable_v2_behavior()
  tf.test.main()
