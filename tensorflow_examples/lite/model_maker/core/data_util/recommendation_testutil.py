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
import tempfile
from unittest import mock

import tensorflow as tf

from tensorflow_examples.lite.model_maker.core.data_util import recommendation_config
from tensorflow_examples.lite.model_maker.third_party.recommendation.ml.data import example_generation_movielens as gen

MOVIE_SIZE = 101
RATING_SIZE = 5000
USER_SIZE = 50

TRAIN_SIZE = 4455
TEST_SIZE = 495
VOCAB_SIZE = 101
MAX_ITEM_ID = 999


def _generate_fake_data(data_dir):
  """Generates fake data to files.

  It generates 3 files.
  - movies.dat: movies data, with format per line:
               MovieID::Title::Genres.
  - users.dat: users data, with format per line:
               UserID::Gender::Age::Occupation::Zip-code.
  - ratings.dat: movie ratings by users, with format per line:
               UserID::MovieID::Rating::Timestamp
  It aligns with movielens dataset. IDs start from 1, and 0 is reserved for OOV.

  Args:
    data_dir: str, dir name to generate dataset.
  """
  if not tf.io.gfile.exists(data_dir):
    tf.io.gfile.makedirs(data_dir)

  # Movies:
  # MovieID::Title::Genres
  movies = ['{i}::title{i}::genre1|genre2'.format(i=i) for i in range(1, 101)]
  movies.append('999::title999::genere10')  # Add a movie with a larger id.
  movie_file = os.path.join(data_dir, 'movies.dat')
  _write_file_by_lines(movies, movie_file)

  # Users:
  # UserID::Gender::Age::Occupation::Zip-code
  users = ['{user}::F::0::0::00000'.format(user=user) for user in range(1, 51)]
  user_file = os.path.join(data_dir, 'users.dat')
  _write_file_by_lines(users, user_file)

  # Ratings:
  # UserID::MovieID::Rating::Timestamp
  ratings = []
  for user in range(1, 51):
    ratings += [
        '{user}::{movie}::5::{timestamp}'.format(
            user=user, movie=movie, timestamp=978000000 + 1)
        for movie in range(1, 101)
    ]
  rating_file = os.path.join(data_dir, 'ratings.dat')
  _write_file_by_lines(ratings, rating_file)


def _write_file_by_lines(data, filename):
  """Writes data to file line per line."""
  with tf.io.gfile.GFile(filename, 'w') as f:
    for line in data:
      f.write(str(line))
      f.write('\n')


def setup_fake_testdata(obj):
  """Setup fake testdata folder.

  This function creates new attrs:
  - test_tempdir: temporary dir (optional), if not exists.
  - download_dir: datasets dir for downloaded zip.
  - dataset_dir: extracted dir for movielens data.

  Args:
    obj: object, usually TestCase instance's self or cls.
  """
  if not hasattr(obj, 'test_tempdir'):
    obj.test_tempdir = tempfile.mkdtemp()
  obj.download_dir = os.path.join(obj.test_tempdir, 'download')
  obj.dataset_dir = os.path.join(obj.download_dir, 'fake_movielens')
  _generate_fake_data(obj.dataset_dir)


def patch_download_and_extract_data(data_dir):
  """Patch download and extract data for testing.

  The common usage is to generate data loader:

  with patch_download_and_extract_data(movielens_dir):
    train_loader = RecommendationDataLoader.from_movielens(
        generated_dir, 'train', test_tempdir)

  Args:
    data_dir: str, path to data dir.

  Returns:
    mocked context.
  """

  def side_effect(*args, **kwargs):
    del args, kwargs
    tf.compat.v1.logging.info('Use patched dataset dir: %s', data_dir)
    return data_dir

  return mock.patch.object(
      gen, 'download_and_extract_data', side_effect=side_effect)


def get_input_spec(encoder_type='cnn') -> recommendation_config.InputSpec:
  """Gets input spec (for test).

  Input spec defines how the input features are extracted.

  Args:
    encoder_type: str. Case-insensitive {'CNN', 'LSTM', 'BOW'}.

  Returns:
    InputSpec.
  """
  etype = encoder_type.upper()
  if etype not in {'CNN', 'LSTM', 'BOW'}:
    raise ValueError('Not support encoder_type: {}'.format(etype))

  return recommendation_config.InputSpec(
      activity_feature_groups=[
          # Group #1: defines how features are grouped in the first Group.
          dict(
              features=[
                  # First feature.
                  dict(
                      feature_name='context_movie_id',  # Feature name
                      feature_type='INT',  # Feature type
                      vocab_size=3953,  # ID size (number of IDs)
                      embedding_dim=8,  # Projected feature embedding dim
                      feature_length=10,  # History length of 10.
                  ),
                  # Maybe more features...
              ],
              encoder_type='CNN',  # CNN encoder (e.g. CNN, LSTM, BOW)
          ),
          # Maybe more groups...
      ],
      label_feature=dict(
          feature_name='label_movie_id',  # Label feature name
          feature_type='INT',  # Label type
          vocab_size=3953,  # Label size (number of classes)
          embedding_dim=8,  # Label embedding demension
          feature_length=1,  # Exactly 1 label
      ),
  )


def get_model_hparams() -> recommendation_config.ModelHParams:
  """Gets model hparams (for test).

  ModelHParams defines the model architecture.

  Returns:
    ModelHParams.
  """
  return recommendation_config.ModelHParams(
      hidden_layer_dims=[32, 32],  # Hidden layers dimension.
      eval_top_k=[1, 5],  # Eval top 1 and top 5.
      conv_num_filter_ratios=[2, 4],  # For CNN encoder, conv filter mutipler.
      conv_kernel_size=16,  # For CNN encoder, base kernel size.
      lstm_num_units=16,  # For LSTM/RNN, num units.
      num_predictions=10,  # Number of output predictions. Select top 10.
  )
