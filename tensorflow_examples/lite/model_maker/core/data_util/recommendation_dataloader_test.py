# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for Recommendation dataloader."""

import os

import tensorflow.compat.v2 as tf
from tensorflow_examples.lite.model_maker.core.data_util import recommendation_dataloader as _dl
from tensorflow_examples.lite.model_maker.core.data_util import recommendation_testutil as _testutil


class RecommendationDataLoaderTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    _testutil.setup_fake_testdata(self)

  def test_prepare_movielens_datasets(self):
    loader = _dl.RecommendationDataLoader
    with _testutil.patch_download_and_extract_data(self.movielens_dir):
      stats = loader._prepare_movielens_datasets(
          self.test_tempdir, self.generated_dir, 'train.tfrecord',
          'test.tfrecord', 'movie_vocab.json', 'meta.json')
    self.assertDictContainsSubset(
        {
            'train_file': os.path.join(self.generated_dir, 'train.tfrecord'),
            'test_file': os.path.join(self.generated_dir, 'test.tfrecord'),
            'vocab_file': os.path.join(self.generated_dir, 'movie_vocab.json'),
            'train_size': _testutil.TRAIN_SIZE,
            'test_size': _testutil.TEST_SIZE,
            'vocab_size': _testutil.VOCAB_SIZE,
        }, stats)

    self.assertTrue(os.path.exists(self.movielens_dir))
    self.assertGreater(len(os.listdir(self.movielens_dir)), 0)

    meta_file = os.path.join(self.generated_dir, 'meta.json')
    self.assertTrue(os.path.exists(meta_file))

  def test_from_movielens(self):
    with _testutil.patch_download_and_extract_data(self.movielens_dir):
      train_loader = _dl.RecommendationDataLoader.from_movielens(
          self.generated_dir, 'train', self.test_tempdir)
      test_loader = _dl.RecommendationDataLoader.from_movielens(
          self.generated_dir, 'test', self.test_tempdir)

    self.assertEqual(len(train_loader), _testutil.TRAIN_SIZE)
    self.assertIsNotNone(train_loader._dataset)

    self.assertEqual(len(test_loader), _testutil.TEST_SIZE)
    self.assertIsNotNone(test_loader._dataset)

  def test_split(self):
    with _testutil.patch_download_and_extract_data(self.movielens_dir):
      test_loader = _dl.RecommendationDataLoader.from_movielens(
          self.generated_dir, 'test', self.test_tempdir)
    test0, test1 = test_loader.split(0.1)
    expected_size0 = int(0.1 * _testutil.TEST_SIZE)
    expected_size1 = _testutil.TEST_SIZE - expected_size0
    self.assertEqual(len(test0), expected_size0)
    self.assertIsNotNone(test0._dataset)

    self.assertEqual(len(test1), expected_size1)
    self.assertIsNotNone(test1._dataset)

  def test_load_vocab_and_item_size(self):
    with _testutil.patch_download_and_extract_data(self.movielens_dir):
      test_loader = _dl.RecommendationDataLoader.from_movielens(
          self.generated_dir, 'test', self.test_tempdir)
    vocab, item_size = test_loader.load_vocab_and_item_size()
    self.assertEqual(len(vocab), _testutil.VOCAB_SIZE)
    self.assertEqual(item_size, _testutil.ITEM_SIZE)

  def test_gen_dataset(self):
    with _testutil.patch_download_and_extract_data(self.movielens_dir):
      test_loader = _dl.RecommendationDataLoader.from_movielens(
          self.generated_dir, 'test', self.test_tempdir)
    ds = test_loader.gen_dataset(10, is_training=False)
    self.assertIsInstance(ds, tf.data.Dataset)


if __name__ == '__main__':
  tf.test.main()
