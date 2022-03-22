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
"""Tests for personalized recommendation model."""
import tensorflow as tf

from configs import input_config_generated_pb2 as input_config_pb2
from configs import model_config as model_config_class
from model import recommendation_model


class RecommendationModelTest(tf.test.TestCase):

  def _create_test_input_config(self,
                                encoder_type: input_config_pb2.EncoderType):
    """Generate test input_config_pb2.InputConfig proto."""
    feature_context_movie_id = input_config_pb2.Feature(
        feature_name='context_movie_id',
        feature_type=input_config_pb2.FeatureType.INT,
        vocab_size=20,
        embedding_dim=4)
    feature_context_movie_rating = input_config_pb2.Feature(
        feature_name='context_movie_rating',
        feature_type=input_config_pb2.FeatureType.FLOAT)
    feature_group_1 = input_config_pb2.FeatureGroup(
        features=[feature_context_movie_id, feature_context_movie_rating],
        encoder_type=encoder_type)

    feature_label = input_config_pb2.Feature(
        feature_name='label_movie_id',
        feature_type=input_config_pb2.FeatureType.INT,
        vocab_size=20,
        embedding_dim=4)

    input_config = input_config_pb2.InputConfig(
        activity_feature_groups=[feature_group_1],
        label_feature=feature_label)
    return input_config

  def _create_test_model_config(self):
    return model_config_class.ModelConfig(
        hidden_layer_dims=[8, 4],
        eval_top_k=[1, 5],
        conv_num_filter_ratios=[1, 2],
        conv_kernel_size=2,
        lstm_num_units=16)

  def test_model_train_bow(self):
    input_config = self._create_test_input_config(
        input_config_pb2.EncoderType.BOW)
    model_config = self._create_test_model_config()
    test_model = recommendation_model.RecommendationModel(
        input_config=input_config, model_config=model_config)
    batch_size = 4
    input_context_movie_id = tf.keras.layers.Input(
        shape=(10,),
        dtype=tf.int32,
        batch_size=batch_size,
        name='context_movie_id')
    input_context_movie_rating = tf.keras.layers.Input(
        shape=(10,),
        dtype=tf.float32,
        batch_size=batch_size,
        name='context_movie_rating')
    input_label_movie_id = tf.keras.layers.Input(
        shape=(1,),
        dtype=tf.int32,
        batch_size=batch_size,
        name='label_movie_id')
    inputs = {
        'context_movie_id': input_context_movie_id,
        'context_movie_rating': input_context_movie_rating,
        'label_movie_id': input_label_movie_id
    }
    logits = test_model(inputs)
    self.assertAllEqual([batch_size, 20], logits.shape.as_list())

  def test_model_train_cnn(self):
    input_config = self._create_test_input_config(
        input_config_pb2.EncoderType.CNN)
    model_config = self._create_test_model_config()
    test_model = recommendation_model.RecommendationModel(
        input_config=input_config, model_config=model_config)
    batch_size = 4
    input_context_movie_id = tf.keras.layers.Input(
        shape=(10,),
        dtype=tf.int32,
        batch_size=batch_size,
        name='context_movie_id')
    input_context_movie_rating = tf.keras.layers.Input(
        shape=(10,),
        dtype=tf.float32,
        batch_size=batch_size,
        name='context_movie_rating')
    input_label_movie_id = tf.keras.layers.Input(
        shape=(1,),
        dtype=tf.int32,
        batch_size=batch_size,
        name='label_movie_id')
    inputs = {
        'context_movie_id': input_context_movie_id,
        'context_movie_rating': input_context_movie_rating,
        'label_movie_id': input_label_movie_id
    }
    logits = test_model(inputs)
    self.assertAllEqual([batch_size, 20], logits.shape.as_list())

  def test_model_train_lstm(self):
    input_config = self._create_test_input_config(
        input_config_pb2.EncoderType.LSTM)
    model_config = self._create_test_model_config()
    test_model = recommendation_model.RecommendationModel(
        input_config=input_config, model_config=model_config)
    batch_size = 4
    input_context_movie_id = tf.keras.layers.Input(
        shape=(10,),
        dtype=tf.int32,
        batch_size=batch_size,
        name='context_movie_id')
    input_context_movie_rating = tf.keras.layers.Input(
        shape=(10,),
        dtype=tf.float32,
        batch_size=batch_size,
        name='context_movie_rating')
    input_label_movie_id = tf.keras.layers.Input(
        shape=(1,),
        dtype=tf.int32,
        batch_size=batch_size,
        name='label_movie_id')
    inputs = {
        'context_movie_id': input_context_movie_id,
        'context_movie_rating': input_context_movie_rating,
        'label_movie_id': input_label_movie_id
    }
    logits = test_model(inputs)
    self.assertAllEqual([batch_size, 20], logits.shape.as_list())


if __name__ == '__main__':
  tf.test.main()
