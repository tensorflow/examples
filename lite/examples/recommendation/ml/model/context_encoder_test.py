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
"""Tests for context_encoder."""
import tensorflow as tf

from configs import input_config_generated_pb2 as input_config_pb2
from configs import model_config as model_config_class
from model import context_encoder


class ContextEncoderTest(tf.test.TestCase):

  def _create_test_feature_group(self,
                                 encoder_type: input_config_pb2.EncoderType):
    """Prepare test feature group."""
    feature_context_movie_id = input_config_pb2.Feature(
        feature_name='context_movie_id',
        feature_type=input_config_pb2.FeatureType.INT,
        vocab_size=3952,
        embedding_dim=4)
    feature_context_movie_rating = input_config_pb2.Feature(
        feature_name='context_movie_rating',
        feature_type=input_config_pb2.FeatureType.FLOAT)
    return input_config_pb2.FeatureGroup(
        features=[feature_context_movie_id, feature_context_movie_rating],
        encoder_type=encoder_type)

  def _create_test_input_config(self):
    """Generate test input_config_pb2.InputConfig proto."""
    feature_group_1 = self._create_test_feature_group(
        encoder_type=input_config_pb2.EncoderType.BOW)

    feature_context_movie_genre = input_config_pb2.Feature(
        feature_name='context_movie_genre',
        feature_type=input_config_pb2.FeatureType.STRING,
        vocab_name='movie_genre_vocab.txt',
        vocab_size=19,
        embedding_dim=3)
    feature_group_2 = input_config_pb2.FeatureGroup(
        features=[feature_context_movie_genre],
        encoder_type=input_config_pb2.EncoderType.BOW)

    feature_label = input_config_pb2.Feature(
        feature_name='label_movie_id',
        feature_type=input_config_pb2.FeatureType.INT,
        vocab_size=3952,
        embedding_dim=4)

    input_config = input_config_pb2.InputConfig(
        activity_feature_groups=[feature_group_1, feature_group_2],
        label_feature=feature_label)
    return input_config

  def _create_test_model_config(self):
    return model_config_class.ModelConfig(
        hidden_layer_dims=[8, 4],
        eval_top_k=[1, 5],
        conv_num_filter_ratios=[1, 2],
        conv_kernel_size=2,
        lstm_num_units=16)

  def test_feature_group_encoder_bow(self):
    feature_group = self._create_test_feature_group(
        encoder_type=input_config_pb2.EncoderType.BOW)
    model_config = self._create_test_model_config()
    feature_group_encoder = context_encoder.FeatureGroupEncoder(
        feature_group, model_config, final_embedding_dim=4)
    input_context_movie_id = tf.constant([[1, 0, 0], [1, 2, 0]])
    input_context_movie_rating = tf.constant([[1.0, 0.0, 0.0], [2.0, 3.0, 0.0]])
    input_context = {
        'context_movie_id': input_context_movie_id,
        'context_movie_rating': input_context_movie_rating
    }
    feature_group_embedding = feature_group_encoder(input_context)
    self.assertAllEqual([2, 5], list(feature_group_embedding.shape))

  def test_feature_group_encoder_cnn(self):
    feature_group = self._create_test_feature_group(
        encoder_type=input_config_pb2.EncoderType.CNN)
    model_config = self._create_test_model_config()
    feature_group_encoder = context_encoder.FeatureGroupEncoder(
        feature_group, model_config, final_embedding_dim=4)
    input_context_movie_id = tf.constant([[1, 0, 0], [1, 2, 0]])
    input_context_movie_rating = tf.constant([[1.0, 0.0, 0.0], [2.0, 3.0, 0.0]])
    input_context = {
        'context_movie_id': input_context_movie_id,
        'context_movie_rating': input_context_movie_rating
    }
    feature_group_embedding = feature_group_encoder(input_context)
    self.assertAllEqual([2, 8], list(feature_group_embedding.shape))

  def test_feature_group_encoder_lstm(self):
    feature_group = self._create_test_feature_group(
        encoder_type=input_config_pb2.EncoderType.LSTM)
    model_config = self._create_test_model_config()
    feature_group_encoder = context_encoder.FeatureGroupEncoder(
        feature_group, model_config, final_embedding_dim=4)
    input_context_movie_id = tf.constant([[1, 0, 0], [1, 2, 0]])
    input_context_movie_rating = tf.constant([[1.0, 0.0, 0.0], [2.0, 3.0, 0.0]])
    input_context = {
        'context_movie_id': input_context_movie_id,
        'context_movie_rating': input_context_movie_rating
    }
    feature_group_embedding = feature_group_encoder(input_context)
    self.assertAllEqual([2, 16], list(feature_group_embedding.shape))

  def test_context_encoder(self):
    input_config = self._create_test_input_config()
    model_config = self._create_test_model_config()
    input_context_encoder = context_encoder.ContextEncoder(
        input_config=input_config, model_config=model_config)
    input_context_movie_id = tf.constant([[1, 0, 0], [1, 2, 0]])
    input_context_movie_rating = tf.constant([[1.0, 0.0, 0.0], [2.0, 3.0, 0.0]])
    input_context_movie_genre = tf.constant([[1, 2, 2, 4, 3], [1, 1, 2, 2, 3]])
    input_context = {
        'context_movie_id': input_context_movie_id,
        'context_movie_rating': input_context_movie_rating,
        'context_movie_genre': input_context_movie_genre
    }
    context_embedding = input_context_encoder(input_context)
    self.assertAllEqual([2, 4], list(context_embedding.shape))

if __name__ == '__main__':
  tf.test.main()
