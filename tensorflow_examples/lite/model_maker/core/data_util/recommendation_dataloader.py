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
import functools
import json
import os

import tensorflow as tf

from tensorflow_examples.lite.model_maker.core import file_util
from tensorflow_examples.lite.model_maker.core.api import mm_export
from tensorflow_examples.lite.model_maker.core.data_util import dataloader
from tensorflow_examples.lite.model_maker.core.data_util import recommendation_config
from tensorflow_examples.lite.model_maker.third_party.recommendation.ml.data import example_generation_movielens as _gen
from tensorflow_examples.lite.model_maker.third_party.recommendation.ml.model import input_pipeline
from tensorflow_examples.lite.model_maker.third_party.recommendation.ml.model import utils


@mm_export('recommendation.DataLoader')
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
                  drop_remainder=True,
                  total_steps=None):
    """Generates dataset, and overwrites default drop_remainder = True."""
    ds = super(RecommendationDataLoader, self).gen_dataset(
        batch_size=batch_size,
        is_training=is_training,
        shuffle=shuffle,
        input_pipeline_context=input_pipeline_context,
        preprocess=preprocess,
        drop_remainder=drop_remainder,
    )
    # TODO(tianlin): Consider to move the num_batches below to the super class.
    # Calculate steps by train data, if it is not set.
    if total_steps:
      num_batches = total_steps
    else:
      num_batches = self._size // batch_size
    return ds.take(num_batches)

  def split(self, fraction):
    return self._split(fraction, self.vocab)

  @classmethod
  def load_vocab(cls, vocab_file) -> collections.OrderedDict:
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

  @classmethod
  def _read_dataset(cls, data_filepattern: str,
                    input_spec: recommendation_config.InputSpec,
                    vocab_file_dir: str) -> tf.data.Dataset:
    features_and_vocabs_by_name = input_pipeline.get_features_and_vocabs_by_name(
        input_spec, vocab_file_dir)
    if not input_spec.HasField('label_feature'):
      raise ValueError('Field label_feature is required.')
    input_files = utils.GetShardFilenames(data_filepattern)
    d = tf.data.TFRecordDataset(input_files)
    d = d.shuffle(len(input_files))
    decode_fn = functools.partial(
        input_pipeline.decode_example,
        features_and_vocabs_by_name=features_and_vocabs_by_name,
        label_feature_name=input_spec.label_feature.feature_name)
    return d.map(decode_fn, num_parallel_calls=tf.data.AUTOTUNE)

  @classmethod
  def download_and_extract_movielens(cls, download_dir):
    """Downloads and extracts movielens dataset, then returns extracted dir."""
    return _gen.download_and_extract_data(download_dir)

  @classmethod
  def generate_movielens_dataset(
      cls,
      data_dir,
      generated_examples_dir=None,
      train_filename='train_movielens_1m.tfrecord',
      test_filename='test_movielens_1m.tfrecord',
      vocab_filename='movie_vocab.json',
      meta_filename='meta.json',
      min_timeline_length=3,
      max_context_length=10,
      max_context_movie_genre_length=10,
      min_rating=None,
      train_data_fraction=0.9,
      build_vocabs=True,
  ):
    """Generate movielens dataset, and returns a dict contains meta.

    Args:
      data_dir: str, path to dataset containing (unzipped) text data.
      generated_examples_dir: str, path to generate preprocessed examples.
        (default: same as data_dir)
      train_filename: str, generated file name for training data.
      test_filename: str, generated file name for test data.
      vocab_filename: str, generated file name for vocab data.
      meta_filename: str, generated file name for meta data.
      min_timeline_length: int, min timeline length to split train/eval set.
      max_context_length: int, max context length as one input.
      max_context_movie_genre_length: int, max context length of movie genre as
        one input.
      min_rating: int or None, include examples with min rating.
      train_data_fraction: float, percentage of training data [0.0, 1.0].
      build_vocabs: boolean, whether to build vocabs.

    Returns:
      Dict, metadata for the movielens dataset. Containing keys:
        `train_file`, `train_size`, `test_file`, `test_size`, vocab_file`,
        `vocab_size`, etc.
    """
    if not generated_examples_dir:
      # By default, set generated examples dir to data_dir
      generated_examples_dir = data_dir
    train_file = os.path.join(generated_examples_dir, train_filename)
    test_file = os.path.join(generated_examples_dir, test_filename)
    meta_file = os.path.join(generated_examples_dir, meta_filename)
    # Create dataset and meta, only if they are not existed.
    if not all([os.path.exists(f) for f in (train_file, test_file, meta_file)]):
      stats = _gen.generate_datasets(
          data_dir,
          output_dir=generated_examples_dir,
          min_timeline_length=min_timeline_length,
          max_context_length=max_context_length,
          max_context_movie_genre_length=max_context_movie_genre_length,
          min_rating=min_rating,
          build_vocabs=build_vocabs,
          train_data_fraction=train_data_fraction,
          train_filename=train_filename,
          test_filename=test_filename,
          vocab_filename=vocab_filename,
      )
      file_util.write_json_file(meta_file, stats)
    meta = file_util.load_json_file(meta_file)
    return meta

  @classmethod
  def get_num_classes(cls, meta) -> int:
    """Gets number of classes.

    0 is reserved. Number of classes is Max Id + 1, e.g., if Max Id = 100,
    then classes are [0, 100], that is 101 classes in total.

    Args:
      meta: dict, containing meta['vocab_max_id'].

    Returns:
      Number of classes.
    """
    return meta['vocab_max_id'] + 1

  @classmethod
  def from_movielens(cls,
                     data_dir,
                     data_tag,
                     input_spec: recommendation_config.InputSpec,
                     generated_examples_dir=None,
                     min_timeline_length=3,
                     max_context_length=10,
                     max_context_movie_genre_length=10,
                     min_rating=None,
                     train_data_fraction=0.9,
                     build_vocabs=True,
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
      data_dir: str, path to dataset containing (unzipped) text data.
      data_tag: str, specify dataset in {'train', 'test'}.
      input_spec: InputSpec, specify data format for input and embedding.
      generated_examples_dir: str, path to generate preprocessed examples.
        (default: same as data_dir)
      min_timeline_length: int, min timeline length to split train/eval set.
      max_context_length: int, max context length as one input.
      max_context_movie_genre_length: int, max context length of movie genre as
        one input.
      min_rating: int or None, include examples with min rating.
      train_data_fraction: float, percentage of training data [0.0, 1.0].
      build_vocabs: boolean, whether to build vocabs.
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
    if not generated_examples_dir:
      # By default, set generated examples dir to data_dir
      generated_examples_dir = data_dir
    meta = cls.generate_movielens_dataset(
        data_dir,
        generated_examples_dir,
        train_filename=train_filename,
        test_filename=test_filename,
        vocab_filename=vocab_filename,
        meta_filename=meta_filename,
        min_timeline_length=min_timeline_length,
        max_context_length=max_context_length,
        max_context_movie_genre_length=max_context_movie_genre_length,
        min_rating=min_rating,
        train_data_fraction=train_data_fraction,
        build_vocabs=build_vocabs,
    )
    vocab = cls.load_vocab(meta['vocab_file'])
    if data_tag == 'train':
      ds = cls._read_dataset(meta['train_file'], input_spec,
                             generated_examples_dir)
      return cls(ds, meta['train_size'], vocab)
    elif data_tag == 'test':
      ds = cls._read_dataset(meta['test_file'], input_spec,
                             generated_examples_dir)
      return cls(ds, meta['test_size'], vocab)
