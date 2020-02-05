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

import collections
import json
import os

import numpy as np
import tensorflow as tf # TF2
from tensorflow_examples.lite.model_customization.core.data_util import dataloader
import tensorflow_examples.lite.model_customization.core.task.model_spec as ms


class DataLoaderTest(tf.test.TestCase):

  def setUp(self):
    super(DataLoaderTest, self).setUp()
    self.model_spec = ms.AverageWordVecModelSpec(seq_len=4)

  def test_split(self):
    ds = tf.data.Dataset.from_tensor_slices([[0, 1], [1, 1], [0, 0], [1, 0]])
    data = dataloader.DataLoader(ds, 4)
    train_data, test_data = data.split(0.5, shuffle=False)

    self.assertEqual(train_data.size, 2)
    for i, elem in enumerate(train_data.dataset):
      self.assertTrue((elem.numpy() == np.array([i, 1])).all())

    self.assertEqual(test_data.size, 2)
    for i, elem in enumerate(test_data.dataset):
      self.assertTrue((elem.numpy() == np.array([i, 0])).all())

  def _get_tfrecord_file(self):
    tfrecord_file = os.path.join(self.get_temp_dir(), 'tmp.tfrecord')
    writer = tf.io.TFRecordWriter(tfrecord_file)
    input_ids = tf.train.Int64List(value=[0, 1, 2, 3])
    label_ids = tf.train.Int64List(value=[0])
    features = collections.OrderedDict()
    features['input_ids'] = tf.train.Feature(int64_list=input_ids)
    features['label_ids'] = tf.train.Feature(int64_list=label_ids)
    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())
    writer.close()
    return tfrecord_file

  def _get_meta_data_file(self):
    meta_data_file = os.path.join(self.get_temp_dir(), 'tmp_meta_data')
    meta_data = {'size': 1, 'num_classes': 1, 'index_to_label': ['0']}
    with tf.io.gfile.GFile(meta_data_file, 'w') as f:
      json.dump(meta_data, f)
    return meta_data_file

  def test_load(self):
    tfrecord_file = self._get_tfrecord_file()
    meta_data_file = self._get_meta_data_file()
    dataset, meta_data = dataloader.load(tfrecord_file, meta_data_file,
                                         self.model_spec)
    for i, (input_ids, label_ids) in enumerate(dataset):
      self.assertEqual(i, 0)
      self.assertTrue((input_ids.numpy() == [0, 1, 2, 3]).all())
      self.assertTrue((label_ids.numpy() == [0]).all())
    self.assertEqual(meta_data['size'], 1)
    self.assertEqual(meta_data['num_classes'], 1)
    self.assertEqual(meta_data['index_to_label'], ['0'])

  def test_get_cache_filenames(self):
    tfrecord_file, meta_data_file, prefix = dataloader.get_cache_filenames(
        cache_dir='/tmp', model_spec=self.model_spec, data_name='train')
    self.assertTrue(tfrecord_file.startswith(prefix))
    self.assertTrue(meta_data_file.startswith(prefix))

    _, _, new_dir_prefix = dataloader.get_cache_filenames(
        cache_dir='/tmp1', model_spec=self.model_spec, data_name='train')
    self.assertNotEqual(new_dir_prefix, prefix)

    _, _, new_model_spec_prefix = dataloader.get_cache_filenames(
        cache_dir='/tmp',
        model_spec=ms.AverageWordVecModelSpec(seq_len=8),
        data_name='train')
    self.assertNotEqual(new_model_spec_prefix, prefix)

    _, _, new_data_name_prefix = dataloader.get_cache_filenames(
        cache_dir='/tmp', model_spec=self.model_spec, data_name='test')
    self.assertNotEqual(new_data_name_prefix, prefix)


if __name__ == '__main__':
  assert tf.__version__.startswith('2')
  tf.test.main()
