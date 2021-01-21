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
import collections
import os

import tensorflow.compat.v2 as tf
from tensorflow_examples.lite.model_maker.core.data_util import recommendation_dataloader as _dl
from tensorflow_examples.lite.model_maker.core.data_util import recommendation_testutil as _testutil


class RecommendationDataLoaderTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    _testutil.setup_fake_testdata(self)

  def test_download_and_extract_data(self):
    with _testutil.patch_download_and_extract_data(self.dataset_dir) as fn:
      out_dir = _dl.RecommendationDataLoader.download_and_extract_movielens(
          self.download_dir)
      fn.called_once_with(self.download_dir)
      self.assertEqual(out_dir, self.dataset_dir)

  def test_generate_movielens_examples(self):
    loader = _dl.RecommendationDataLoader
    gen_dir = os.path.join(self.dataset_dir, 'generated_examples')
    stats = loader._generate_movielens_examples(self.dataset_dir, gen_dir,
                                                'train.tfrecord',
                                                'test.tfrecord',
                                                'movie_vocab.json', 'meta.json')
    self.assertDictContainsSubset(
        {
            'train_file': os.path.join(gen_dir, 'train.tfrecord'),
            'test_file': os.path.join(gen_dir, 'test.tfrecord'),
            'vocab_file': os.path.join(gen_dir, 'movie_vocab.json'),
            'train_size': _testutil.TRAIN_SIZE,
            'test_size': _testutil.TEST_SIZE,
            'vocab_size': _testutil.VOCAB_SIZE,
        }, stats)

    self.assertTrue(os.path.exists(gen_dir))
    self.assertGreater(len(os.listdir(gen_dir)), 0)

    meta_file = os.path.join(gen_dir, 'meta.json')
    self.assertTrue(os.path.exists(meta_file))

  def test_from_movielens(self):
    train_loader = _dl.RecommendationDataLoader.from_movielens(
        self.dataset_dir, 'train')
    test_loader = _dl.RecommendationDataLoader.from_movielens(
        self.dataset_dir, 'test')

    self.assertEqual(len(train_loader), _testutil.TRAIN_SIZE)
    self.assertIsNotNone(train_loader._dataset)
    self.assertIsInstance(train_loader.vocab, collections.OrderedDict)
    self.assertEqual(len(train_loader.vocab), _testutil.VOCAB_SIZE)
    self.assertEqual(train_loader.max_vocab_id, _testutil.MAX_ITEM_ID)

    self.assertEqual(len(test_loader), _testutil.TEST_SIZE)
    self.assertIsNotNone(test_loader._dataset)
    self.assertEqual(len(test_loader.vocab), _testutil.VOCAB_SIZE)
    self.assertIsInstance(test_loader.vocab, collections.OrderedDict)
    self.assertEqual(test_loader.max_vocab_id, _testutil.MAX_ITEM_ID)

  def test_split(self):
    test_loader = _dl.RecommendationDataLoader.from_movielens(
        self.dataset_dir, 'test')
    test0, test1 = test_loader.split(0.1)
    expected_size0 = int(0.1 * _testutil.TEST_SIZE)
    expected_size1 = _testutil.TEST_SIZE - expected_size0
    self.assertEqual(len(test0), expected_size0)
    self.assertIsNotNone(test0._dataset)

    self.assertEqual(len(test1), expected_size1)
    self.assertIsNotNone(test1._dataset)

  def test_gen_dataset(self):
    test_loader = _dl.RecommendationDataLoader.from_movielens(
        self.dataset_dir, 'test')
    ds = test_loader.gen_dataset(10, is_training=False)
    self.assertIsInstance(ds, tf.data.Dataset)


if __name__ == '__main__':
  tf.test.main()
