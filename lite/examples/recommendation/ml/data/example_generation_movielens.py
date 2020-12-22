# Lint as: python3
#   Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""Prepare TF.Examples for on-device recommendation model.

Following functions are included: 1) downloading raw data 2) processing to user
activity sequence and splitting to train/test data 3) convert to TF.Examples
and write in output location.

More information about the movielens dataset can be found here:
https://grouplens.org/datasets/movielens/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import os

from absl import app
from absl import flags
import pandas as pd
import tensorflow as tf

FLAGS = flags.FLAGS
# Permalinks to download movielens data.
MOVIELENS_1M_URL = "http://files.grouplens.org/datasets/movielens/ml-1m.zip"
MOVIELENS_ZIP_FILENAME = "ml-1m.zip"
MOVIELENS_ZIP_HASH = "a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20"
MOVIELENS_EXTRACTED_DIR = "ml-1m"
RATINGS_FILE_NAME = "ratings.dat"
MOVIES_FILE_NAME = "movies.dat"
RATINGS_DATA_COLUMNS = ["UserID", "MovieID", "Rating", "Timestamp"]
MOVIES_DATA_COLUMNS = ["MovieID", "Title", "Genres"]
OUTPUT_TRAINING_DATA_FILENAME = "train_movielens_1m.tfrecord"
OUTPUT_TESTING_DATA_FILENAME = "test_movielens_1m.tfrecord"
OUTPUT_MOVIE_VOCAB_FILENAME = "movie_vocab.json"
OOV_MOVIE_ID = 0


def define_flags():
  flags.DEFINE_string("data_dir", "/tmp",
                      "Path to download and store movielens data.")
  flags.DEFINE_string("output_dir", None,
                      "Path to the directory of output files.")
  flags.DEFINE_bool("build_movie_vocab", True,
                    "If yes, generate sorted movie vocab.")
  flags.DEFINE_integer("min_timeline_length", 3,
                       "The minimum timeline length to construct examples.")
  flags.DEFINE_integer("max_context_length", 10,
                       "The maximun length of user context history.")


def download_and_extract_data(data_directory, url=MOVIELENS_1M_URL):
  """Download and extract zip containing MovieLens data to a given directory.

  Args:
    data_directory: Local path to extract dataset to.
    url: Direct path to MovieLens dataset .zip file. See constants above for
      examples.

  Returns:
    Downloaded and extracted data file directory.
  """
  path_to_zip = tf.keras.utils.get_file(
      fname=MOVIELENS_ZIP_FILENAME,
      origin=url,
      file_hash=MOVIELENS_ZIP_HASH,
      hash_algorithm="sha256",
      extract=True,
      cache_dir=data_directory)
  extracted_file_dir = os.path.join(
      os.path.dirname(path_to_zip), MOVIELENS_EXTRACTED_DIR)
  return extracted_file_dir


def read_data(data_directory):
  """Read movielens ratings.dat and movies.dat file into dataframe."""
  ratings_df = pd.read_csv(
      os.path.join(data_directory, RATINGS_FILE_NAME),
      sep="::",
      names=RATINGS_DATA_COLUMNS)
  ratings_df["Timestamp"] = ratings_df["Timestamp"].apply(int)
  movies_df = pd.read_csv(
      os.path.join(data_directory, MOVIES_FILE_NAME),
      sep="::",
      names=MOVIES_DATA_COLUMNS)
  return ratings_df, movies_df


def convert_to_timelines(ratings_df):
  """Convert ratings data to user."""
  timelines = collections.defaultdict(list)
  movie_counts = collections.Counter()
  for user_id, movie_id, _, timestamp in ratings_df.values:
    timelines[user_id].append([movie_id, int(timestamp)])
    movie_counts[movie_id] += 1
  # Sort per-user timeline by timestamp
  for (user_id, timeline) in timelines.items():
    timeline.sort(key=lambda x: x[1])
    timelines[user_id] = [movie_id for movie_id, _ in timeline]
  return timelines, movie_counts


def generate_examples_from_timelines(timelines,
                                     min_timeline_len=3,
                                     max_context_len=100):
  """Convert user timelines to tf examples.

  Convert user timelines to tf examples by adding all possible context-label
  pairs in the examples pool.

  Args:
    timelines: the user timelines to process.
    min_timeline_len: minimum length of the user timeline.
    max_context_len: maximum length of context signals.

  Returns:
    train_examples: tf example list for training.
    test_examples: tf example list for testing.
  """
  train_examples = []
  test_examples = []
  for timeline in timelines.values():
    # Skip if timeline is shorter than min_timeline_len.
    if len(timeline) < min_timeline_len:
      continue
    for label_idx in range(1, len(timeline)):
      start_idx = max(0, label_idx - max_context_len)
      context = timeline[start_idx:label_idx]
      # Pad context with out-of-vocab movie id 0.
      while len(context) < max_context_len:
        context.append(OOV_MOVIE_ID)
      label = timeline[label_idx]
      feature = {
          "context":
              tf.train.Feature(int64_list=tf.train.Int64List(value=context)),
          "label":
              tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
      }
      tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
      if label_idx == len(timeline) - 1:
        test_examples.append(tf_example.SerializeToString())
      else:
        train_examples.append(tf_example.SerializeToString())
  return train_examples, test_examples


def write_tfrecords(tf_examples, filename):
  """Writes tf examples to tfrecord file, and returns the count."""
  with tf.io.TFRecordWriter(filename) as file_writer:
    i = 0
    for example in tf_examples:
      file_writer.write(example)
      i += 1
    return i


def generate_sorted_movie_vocab(movies_df, movie_counts):
  """Generate vocabulary for movies, and sort by usage count."""
  vocab_movies = []
  for movie_id, title, genres in movies_df.values:
    count = movie_counts[movie_id] if movie_id in movie_counts else 0
    vocab_movies.append([movie_id, title, genres, count])
  vocab_movies.sort(key=lambda x: x[3], reverse=True)
  return vocab_movies


def write_vocab_json(vocab_movies, filename):
  """Write generated movie vocabulary to specified file."""
  with open(filename, "w", encoding="utf-8") as jsonfile:
    json.dump(vocab_movies, jsonfile, indent=2)


def generate_datasets(data_dir, output_dir, min_timeline_length,
                      max_context_length, build_movie_vocab):
  """Generates train and test datasets as TFRecord, and returns stats."""
  if not tf.io.gfile.exists(data_dir):
    tf.io.gfile.makedirs(data_dir)

  extracted_file_dir = download_and_extract_data(data_directory=data_dir)
  ratings_df, movies_df = read_data(data_directory=extracted_file_dir)
  timelines, movie_counts = convert_to_timelines(ratings_df)
  train_examples, test_examples = generate_examples_from_timelines(
      timelines=timelines,
      min_timeline_len=min_timeline_length,
      max_context_len=max_context_length)

  if not tf.io.gfile.exists(output_dir):
    tf.io.gfile.makedirs(output_dir)
  train_file = os.path.join(output_dir, OUTPUT_TRAINING_DATA_FILENAME)
  train_size = write_tfrecords(tf_examples=train_examples, filename=train_file)
  test_file = os.path.join(output_dir, OUTPUT_TESTING_DATA_FILENAME)
  test_size = write_tfrecords(tf_examples=test_examples, filename=test_file)
  stats = {
      "train_size": train_size,
      "test_size": test_size,
      "train_file": train_file,
      "test_file": test_file,
  }
  if build_movie_vocab:
    vocab_movies = generate_sorted_movie_vocab(
        movies_df=movies_df, movie_counts=movie_counts)
    vocab_file = os.path.join(output_dir, OUTPUT_MOVIE_VOCAB_FILENAME)
    write_vocab_json(vocab_movies=vocab_movies, filename=vocab_file)
    stats.update(vocab_size=len(vocab_movies), vocab_file=vocab_file)
  return stats


def main(_):
  stats = generate_datasets(FLAGS.data_dir, FLAGS.output_dir,
                            FLAGS.min_timeline_length, FLAGS.max_context_length,
                            FLAGS.build_movie_vocab)
  tf.compat.v1.logging.info("Generated dataset: %s", stats)


if __name__ == "__main__":
  define_flags()
  app.run(main)
