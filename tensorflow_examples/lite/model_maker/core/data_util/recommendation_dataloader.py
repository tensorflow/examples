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
"""Recommendation dataloader class."""

import collections
import json
import os

import tensorflow as tf

from tensorflow_examples.lite.model_maker.core import file_util
from tensorflow_examples.lite.model_maker.core.data_util import dataloader
from tensorflow_examples.lite.model_maker.third_party.recommendation.ml.data import example_generation_movielens as _gen
from tensorflow_examples.lite.model_maker.third_party.recommendation.ml.model import recommendation_model_launcher_keras as _launcher


class RecommendationDataLoader(dataloader.DataLoader):
  """Recommendation data loader."""

  def __init__(self, dataset, size, vocab):
    """Init data loader.

    Dataset is tf.data.Dataset of examples, containing:
      for inputs:
      - 'context': int64[], context ids as the input of variable length.
      for outputs:
      - 'label': int64[1], label id to predict.
    where context is controlled by `max_context_length` in generating examples.

    The vocab should be a dict maps `id` to `item`, where:
    - id: int
    - item: a vocab entry. For example, for movielens, the item is a dict:
        {'id': int, 'title': str, 'genres': list[str], 'count': int}

    Args:
      dataset: tf.data.Dataset for recommendation.
      size: int, dataset size.
      vocab: list of dict, each vocab item is described above.
    """
    super(RecommendationDataLoader, self).__init__(dataset, size)
    if not isinstance(vocab, dict):
      raise ValueError('Expect vocab to be a dict, but got: {}'.format(vocab))
    self.vocab = vocab
    self.max_vocab_id = max(self.vocab.keys())  # The max id in the vocab.

  def gen_dataset(self,
                  batch_size=1,
                  is_training=False,
                  shuffle=False,
                  input_pipeline_context=None,
                  preprocess=None,
                  drop_remainder=True):
    """Generates dataset, and overwrites default drop_remainder = True."""
    return super(RecommendationDataLoader, self).gen_dataset(
        batch_size=batch_size,
        is_training=is_training,
        shuffle=shuffle,
        input_pipeline_context=input_pipeline_context,
        preprocess=preprocess,
        drop_remainder=drop_remainder,
    )

  def split(self, fraction):
    return self._split(fraction, self.vocab)

  @classmethod
  def load_vocab(cls, vocab_file):
    """Loads vocab from file.

    The vocab file should be json format of: a list of list[size=4], where the 4
    elements are ordered as:
      [id=int, title=str, genres=str joined with '|', count=int]
    It is generated when preparing movielens dataset.

    Args:
      vocab_file: str, path to vocab file.

    Returns:
      vocab: an OrderedDict maps id to item. Each item represents a movie
         {
           'id': int,
           'title': str,
           'genres': list[str],
           'count': int,
         }
    """
    with tf.io.gfile.GFile(vocab_file) as f:
      vocab_json = json.load(f)
      vocab = collections.OrderedDict()
      for v in vocab_json:
        item = {
            'id': int(v[0]),
            'title': v[1],
            'genres': v[2].split('|'),
            'count': int(v[3]),
        }
        vocab[item['id']] = item
      return vocab

  @staticmethod
  def read_as_dataset(filepattern):
    """Reads file pattern as dataset."""
    dataset = _launcher.InputFn.read_dataset(filepattern)
    return dataset.map(
        _launcher.InputFn.decode_example,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

  @classmethod
  def _prepare_movielens_datasets(cls,
                                  raw_data_dir,
                                  generated_dir,
                                  train_filename,
                                  test_filename,
                                  vocab_filename,
                                  meta_filename,
                                  min_timeline_length=3,
                                  max_context_length=10):
    """Prepare movielens datasets, and returns a dict contains meta."""
    train_file = os.path.join(generated_dir, train_filename)
    test_file = os.path.join(generated_dir, test_filename)
    meta_file = os.path.join(generated_dir, meta_filename)
    # Create dataset and meta, only if they are not existed.
    if not all([os.path.exists(f) for f in (train_file, test_file, meta_file)]):
      stats = _gen.generate_datasets(
          data_dir=raw_data_dir,
          output_dir=generated_dir,
          min_timeline_length=min_timeline_length,
          max_context_length=max_context_length,
          build_movie_vocab=True,
          train_filename=train_filename,
          test_filename=test_filename,
          vocab_filename=vocab_filename,
      )
      file_util.write_json_file(meta_file, stats)
    meta = file_util.load_json_file(meta_file)
    return meta

  @classmethod
  def from_movielens(cls,
                     generated_dir,
                     data_tag,
                     raw_data_dir,
                     min_timeline_length=3,
                     max_context_length=10,
                     train_filename='train_movielens_1m.tfrecord',
                     test_filename='test_movielens_1m.tfrecord',
                     vocab_filename='movie_vocab.json',
                     meta_filename='meta.json'):
    """Generates data loader from movielens dataset.

    The method downloads and prepares dataset, then generates for train/eval.

    For `movielens` data format, see:
    - function `_generate_fake_data` in `recommendation_testutil.py`
    - Or, zip file: http://files.grouplens.org/datasets/movielens/ml-1m.zip

    Args:
      generated_dir: str, path to generate preprocessed examples.
      data_tag: str, specify dataset in {'train', 'test'}.
      raw_data_dir: str, path to download raw data, and unzip.
      min_timeline_length: int, min timeline length to split train/eval set.
      max_context_length: int, max context length as the input.
      train_filename: str, generated file name for training data.
      test_filename: str, generated file name for test data.
      vocab_filename: str, generated file name for vocab data.
      meta_filename: str, generated file name for meta data.

    Returns:
      Data Loader.
    """
    if data_tag not in ('train', 'test'):
      raise ValueError(
          'Expected data_tag is train or test, but got {}'.format(data_tag))
    meta = cls._prepare_movielens_datasets(
        raw_data_dir,
        generated_dir,
        train_filename=train_filename,
        test_filename=test_filename,
        vocab_filename=vocab_filename,
        meta_filename=meta_filename,
        min_timeline_length=min_timeline_length,
        max_context_length=max_context_length)

    vocab = cls.load_vocab(meta['vocab_file'])
    if data_tag == 'train':
      ds = cls.read_as_dataset(meta['train_file'])
      return cls(ds, meta['train_size'], vocab)
    elif data_tag == 'test':
      ds = cls.read_as_dataset(meta['test_file'])
      return cls(ds, meta['test_size'], vocab)
