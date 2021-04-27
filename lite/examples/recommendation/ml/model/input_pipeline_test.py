# Lint as: python3
#   Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for input_pipeline."""
import collections
import os

import tensorflow as tf

from configs import input_config_generated_pb2 as input_config_pb2
from model import input_pipeline

from google.protobuf import text_format

FAKE_MOVIE_GENRE_VOCAB = [
    'UNK', 'Comedy', 'Drama', 'Romance', 'Animation', 'Children'
]

TEST_INPUT_CONFIG = text_format.Parse(
    """
    activity_feature_groups {
      features {
        feature_name: "context_movie_id"
        feature_type: INT
        vocab_size: 3952
        embedding_dim: 32
        feature_length: 5
      }
      features {
        feature_name: "context_movie_rating"
        feature_type: FLOAT
        feature_length: 5
      }
      encoder_type: BOW
    }
    activity_feature_groups {
      features {
        feature_name: "context_movie_genre"
        feature_type: STRING
        vocab_name: "movie_genre_vocab.txt"
        vocab_size: 19
        embedding_dim: 8
        feature_length: 8
      }
      encoder_type: BOW
    }
    label_feature {
      feature_name: "label_movie_id"
      feature_type: INT
      vocab_size: 3952
      embedding_dim: 8
      feature_length: 1
    }
    """, input_config_pb2.InputConfig())

EXAMPLE1 = text_format.Parse(
    """
    features {
        feature {
          key: "context_movie_id"
          value {
            int64_list {
              value: [1, 2, 0, 0, 0]
            }
          }
        }
        feature {
          key: "context_movie_rating"
          value {
            float_list {
              value: [3.5, 4.0, 0.0, 0.0, 0.0]
            }
          }
        }
        feature {
          key: "context_movie_genre"
          value {
            bytes_list {
              value: [
                    "Animation", "Children", "Comedy", "Comedy", "Romance", "UNK", "UNK", "UNK"
                ]
            }
          }
        }
        feature {
          key: "label_movie_id"
          value {
            int64_list {
              value: [3]
            }
          }
        }
      }""", tf.train.Example())


class InputPipelineTest(tf.test.TestCase):

  def _AssertSparseTensorValueEqual(self, a, b):
    self.assertAllEqual(a.indices, b.indices)
    self.assertAllEqual(a.values, b.values)
    self.assertAllEqual(a.dense_shape, b.dense_shape)

  def setUp(self):
    super(InputPipelineTest, self).setUp()
    self.tmp_dir = self.create_tempdir()
    self.test_movie_genre_vocab_file = os.path.join(self.tmp_dir,
                                                    'movie_genre_vocab.txt')
    self.test_input_data_file = os.path.join(self.tmp_dir,
                                             'test_input_data.tfrecord')
    with open(self.test_movie_genre_vocab_file, 'w', encoding='utf-8') as f:
      for item in FAKE_MOVIE_GENRE_VOCAB:
        f.write(item + '\n')
    with tf.io.TFRecordWriter(self.test_input_data_file) as file_writer:
      file_writer.write(EXAMPLE1.SerializeToString())

  def test_features_vocabs_gen(self):
    features_and_vocabs_by_name = (
        input_pipeline.get_features_and_vocabs_by_name(TEST_INPUT_CONFIG,
                                                       self.tmp_dir))
    features_by_name = features_and_vocabs_by_name.features_by_name
    vocabs_by_name = features_and_vocabs_by_name.vocabs_by_name
    self.assertLen(features_by_name.keys(), 3)
    self.assertLen(vocabs_by_name.keys(), 1)
    self.assertAllInSet(
        ['context_movie_id', 'context_movie_rating', 'context_movie_genre'],
        features_by_name.keys())
    self.assertAllInSet(['context_movie_genre'], vocabs_by_name.keys())

  def test_decode_example(self):
    features_and_vocabs_by_name = (
        input_pipeline.get_features_and_vocabs_by_name(TEST_INPUT_CONFIG,
                                                       self.tmp_dir))
    features, label_feature = input_pipeline.decode_example(
        EXAMPLE1.SerializeToString(),
        features_and_vocabs_by_name,
        'label_movie_id')
    expected_context_movie_id = tf.constant([1, 2, 0, 0, 0],
                                            dtype=tf.int32)
    expected_context_movie_genre = tf.constant([4, 5, 1, 1, 3, 0, 0, 0],
                                               dtype=tf.int32)
    expected_context_movie_rating = tf.constant([3.5, 4.0, 0.0, 0.0, 0.0],
                                                dtype=tf.float32)
    self.assertAllEqual(features['context_movie_id'],
                        expected_context_movie_id)
    self.assertAllEqual(features['context_movie_genre'],
                        expected_context_movie_genre)
    self.assertAllEqual(features['context_movie_rating'],
                        expected_context_movie_rating)
    self.assertAllEqual(label_feature, tf.constant([3], dtype=tf.int32))

  def test_get_input_dataset(self):
    dataset = input_pipeline.get_input_dataset(
        data_filepattern=self.test_input_data_file,
        input_config=TEST_INPUT_CONFIG,
        vocab_file_dir=self.tmp_dir,
        batch_size=1)
    dataset = dataset.take(1)
    self.assertCountEqual([
        'context_movie_id', 'context_movie_rating', 'context_movie_genre',
        'label_movie_id'
    ], dataset.element_spec[0].keys())

  def test_get_serving_input_specs(self):
    input_specs = input_pipeline.get_serving_input_specs(TEST_INPUT_CONFIG)
    expected_input_specs = collections.OrderedDict()
    expected_input_specs['context_movie_genre'] = tf.TensorSpec(
        shape=[8], dtype=tf.dtypes.int32, name='context_movie_genre')
    expected_input_specs['context_movie_id'] = tf.TensorSpec(
        shape=[5], dtype=tf.dtypes.int32, name='context_movie_id')
    expected_input_specs['context_movie_rating'] = tf.TensorSpec(
        shape=[5], dtype=tf.dtypes.float32, name='context_movie_rating')
    self.assertAllEqual(input_specs, expected_input_specs)


if __name__ == '__main__':
  tf.test.main()
