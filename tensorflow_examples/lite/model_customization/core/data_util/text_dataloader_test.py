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

import csv
import os
import numpy as np
import tensorflow as tf # TF2

from tensorflow_examples.lite.model_customization.core.data_util import text_dataloader
import tensorflow_examples.lite.model_customization.core.task.model_spec as ms


class TextDataLoaderTest(tf.test.TestCase):
  TEST_LABELS_AND_TEXT = (('pos', 'super good'), ('neg', 'really bad'))

  def _get_folder_path(self):
    folder_path = os.path.join(self.get_temp_dir(), 'random_text_dir')
    if os.path.exists(folder_path):
      return
    os.mkdir(folder_path)

    for label, text in self.TEST_LABELS_AND_TEXT:
      class_subdir = os.path.join(folder_path, label)
      os.mkdir(class_subdir)
      with open(os.path.join(class_subdir, '0.txt'), 'w') as f:
        f.write(text)
    return folder_path

  def _get_csv_file(self):
    csv_file = os.path.join(self.get_temp_dir(), 'tmp.csv')
    if os.path.exists(csv_file):
      return csv_file
    fieldnames = ['text', 'label']
    with open(csv_file, 'w') as f:
      writer = csv.DictWriter(f, fieldnames=fieldnames)
      writer.writeheader()
      for label, text in self.TEST_LABELS_AND_TEXT:
        writer.writerow({'text': text, 'label': label})
    return csv_file

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

  def test_from_csv(self):
    csv_file = self._get_csv_file()
    model_spec = ms.AverageWordVecModelSpec()
    data = text_dataloader.TextClassifierDataLoader.from_csv(
        csv_file,
        text_column='text',
        label_column='label',
        model_spec=model_spec)
    self._test_data(data, model_spec)

  def test_from_folder(self):
    folder_path = self._get_folder_path()
    model_spec = ms.AverageWordVecModelSpec()
    data = text_dataloader.TextClassifierDataLoader.from_folder(
        folder_path, model_spec=model_spec)
    self._test_data(data, model_spec)

  def _test_data(self, data, model_spec):
    self.assertEqual(data.size, 2)
    self.assertEqual(data.num_classes, 2)
    self.assertEqual(data.index_to_label, ['neg', 'pos'])
    for input_ids, label in data.dataset:
      self.assertTrue(label.numpy() == 1 or label.numpy() == 0)
      if label.numpy() == 0:
        actual_input_ids = model_spec.preprocess('really bad')
      else:
        actual_input_ids = model_spec.preprocess('super good')
      self.assertTrue((input_ids.numpy() == actual_input_ids).all())


if __name__ == '__main__':
  assert tf.__version__.startswith('2')
  tf.test.main()
