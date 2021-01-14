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

from tensorflow_examples.lite.model_maker.third_party.recommendation.ml.data import example_generation_movielens as gen

MOVIE_SIZE = 101
RATING_SIZE = 5000
USER_SIZE = 50

TRAIN_SIZE = 4900
TEST_SIZE = 50
VOCAB_SIZE = 101
ITEM_SIZE = 999


def _generate_fake_data(data_dir):
  """Generates fake data to files.

  It generates 3 files.
  - movies.dat: movies data, with format per line:
               MovieID::Title::Genres.
  - users.dat: users data, with format per line:
               UserID::Gender::Age::Occupation::Zip-code.
  - ratings.dat: movie ratings by users, with format per line:
               UserID::MovieID::Rating::Timestamp

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
        '{user}::{movie}::5::978000000'.format(user=user, movie=movie)
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
  - datasets_dir: datasets dir to mock downloaded dir.
  - movielens_dir: extracted dir for movielens data.
  - generated_dir: generated dir after preprocessing movielens data.

  Args:
    obj: object, usually TestCase instance's self or cls.
  """
  if not hasattr(obj, 'test_tempdir'):
    obj.test_tempdir = tempfile.mkdtemp()
  obj.datasets_dir = os.path.join(obj.test_tempdir, 'datasets')
  obj.movielens_dir = os.path.join(obj.datasets_dir, 'fake_movielens')
  _generate_fake_data(obj.movielens_dir)
  obj.generated_dir = os.path.join(obj.test_tempdir, 'generated_data')


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
