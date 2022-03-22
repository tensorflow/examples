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
"""Context encoder layer for on-device personalized recommendation model."""
import math
from typing import Dict

import tensorflow as tf

from configs import input_config_generated_pb2 as input_config_pb2
from configs import model_config as model_config_class


def safe_div(x, y):
  return tf.where(tf.not_equal(y, 0), tf.divide(x, y), tf.zeros_like(x))


class ContextEncoder(tf.keras.layers.Layer):
  """Layer to encode context sequence.

  This encoder layer supports three types: 1) bow: bag of words style averaging
  sequence embeddings. 2) cnn: use convolutional neural network to encode
  sequence. 3) rnn: use recurrent neural network to encode sequence.

  This encoder should be initialized with a predefined embedding layer, encoder
  type and necessary parameters corresponding to the encoder type.
  """

  def __init__(self,
               input_config: input_config_pb2.InputConfig,
               model_config: model_config_class.ModelConfig):
    """Initialize ContextEncoder layer.

    Initialize ContextEncoder layer according to input config and model config,
    setting up feature group encoders and hidden layers.

    Args:
      input_config: The input config (input_config_pb2.InputConfig).
      model_config: The model config (model_config_class.ModelConfig).
    """
    super(ContextEncoder, self).__init__()
    self._input_config = input_config
    self._model_config = model_config
    self._final_embedding_dim = self._input_config.label_feature.embedding_dim
    self._feature_group_encoders = [
        FeatureGroupEncoder(feature_group, self._model_config,
                            self._final_embedding_dim)
        for feature_group in input_config.global_feature_groups
    ]
    self._feature_group_encoders.extend([
        FeatureGroupEncoder(feature_group, self._model_config,
                            self._final_embedding_dim)
        for feature_group in input_config.activity_feature_groups
    ])
    self._final_embedding_dim = self._input_config.label_feature.embedding_dim
    # Set up hidden layers.
    self._hidden_layers = []
    for i, layer_dim in enumerate(self._model_config.hidden_layer_dims):
      self._hidden_layers.append(
          tf.keras.layers.Dense(
              units=layer_dim,
              name='hidden_layer{}'.format(i),
              activation=tf.nn.relu))
    # Append final top layer, dimension of which should equal final embedding
    # dimension.
    self._hidden_layers.append(
        tf.keras.layers.Dense(
            units=self._final_embedding_dim,
            name='proj_layer',
            activation=tf.nn.relu))

  def call(self, input_context: Dict[str, tf.Tensor]) -> tf.Tensor:
    """Call function of the context encoder layer.

    Takes in input context tensor dictionary, generates feature group
    embeddings, concatenate them up and pass them through top hidden layers
    to get final context embedding.

    Args:
      input_context: A dictionary mapping input context feature name to tenors.

    Returns:
      Context encoding.
    """
    input_context = {
        k: tf.expand_dims(v, 0) if v.shape.ndims == 1 else v
        for k, v in input_context.items()
    }
    feature_group_embeddings = [
        feature_group_encoder(input_context)
        for feature_group_encoder in self._feature_group_encoders
    ]
    context_embedding = tf.concat(feature_group_embeddings, -1)
    for hidden_layer in self._hidden_layers:
      context_embedding = hidden_layer(context_embedding)
    return context_embedding


class FeatureGroupEncoder(tf.keras.layers.Layer):
  """Layer to generate encoding for the feature group.

  Layer to generate encoding for the group of features in the feature
  group with encoder type (BOW/CNN/LSTM) specified in the feature group config.
  Embeddings of INT or STRING type features are concatenated first, and FLOAT
  type feature values are appended after. Embedding vector is properly masked.
  """

  def __init__(self,
               feature_group: input_config_pb2.FeatureGroup,
               model_config: model_config_class.ModelConfig,
               final_embedding_dim: int):
    """Initialize FeatureGroupEncoder layer.

    Initialize ContextEncoder layer according to feature group.

    Args:
      feature_group: The feature group that needs to encode.
      model_config: The model config (model_config_class.ModelConfig).
      final_embedding_dim: Dimension of final context embedding.
    """
    super(FeatureGroupEncoder, self).__init__()
    self._feature_group = feature_group
    self._model_config = model_config
    self._final_embedding_dim = final_embedding_dim
    # Prepare embedding layers.
    self._embedding_layer_dict = {}
    self._nonembedding_feature_names = []
    for feature in self._feature_group.features:
      if feature.feature_type in [
          input_config_pb2.FeatureType.INT, input_config_pb2.FeatureType.STRING
      ]:
        assert feature.HasField('vocab_size')
        assert feature.HasField('embedding_dim')
        embedding_layer = tf.keras.layers.Embedding(
            feature.vocab_size,
            feature.embedding_dim,
            embeddings_initializer=tf.keras.initializers.truncated_normal(
                mean=0.0, stddev=1.0 / math.sqrt(feature.embedding_dim)),
            mask_zero=True,
            name=feature.feature_name+'embedding_layer')
        self._embedding_layer_dict[feature.feature_name] = embedding_layer
      else:
        self._nonembedding_feature_names.append(feature.feature_name)
  # Prepare CNN layers.
    self._conv1d_layers = []
    if self._feature_group.encoder_type == input_config_pb2.EncoderType.CNN:
      assert self._model_config.conv_num_filter_ratios
      assert self._model_config.conv_kernel_size
      for ratio in self._model_config.conv_num_filter_ratios:
        self._conv1d_layers.append(
            tf.keras.layers.Conv1D(
                filters=self._final_embedding_dim * ratio,
                kernel_size=self._model_config.conv_kernel_size,
                strides=self._model_config.conv_kernel_size,
                padding='same',
                activation='relu'))
    # Prepare LSTM layer.
    elif self._feature_group.encoder_type == input_config_pb2.EncoderType.LSTM:
      assert self._model_config.lstm_num_units
      self._lstm_layer = tf.keras.layers.LSTM(
          self._model_config.lstm_num_units)

  def call(self, input_context: Dict[str, tf.Tensor]) -> tf.Tensor:
    """Call function of the feature group encoder layer.

    Takes in input context tensor dictionary, generates embedding for features
    in the group, concatenate them up and encode them according to the
    specified encoder type.

    Feature embedding layers will have mask_zero enabled, and generated feature
    embeddings will be masked if any of the feature value is 0. In most
    cases, the mask of all features in the same group should be the same. This
    function treats the input config generically.

    Args:
      input_context: A dictionary mapping input context feature name to tenors.

    Returns:
      Feature group encoding.
    """
    embedding_feature_shape = []
    embeddings = []
    masks = []
    for feature_name, embedding_layer in sorted(
        self._embedding_layer_dict.items()):
      input_feature = input_context[feature_name]
      if isinstance(input_feature, tf.SparseTensor):
        input_feature = tf.sparse.to_dense(input_feature)
      if not embedding_feature_shape:
        embedding_feature_shape = input_feature.shape.as_list()
      assert input_feature.shape.as_list() == embedding_feature_shape

      embeddings.append(embedding_layer(input_feature))
      masks.append(embedding_layer.compute_mask(input_feature))
    # Collect nonembedding feature values.
    for feature_name in self._nonembedding_feature_names:
      input_feature = input_context[feature_name]
      if isinstance(input_feature, tf.SparseTensor):
        input_feature = tf.sparse.to_dense(input_feature)
      embeddings.append(tf.expand_dims(input_feature, -1))
    embedding = tf.concat(embeddings, -1)

    # Set mask_shape with batch dim = 1. Compatible to unknown batch size.
    mask_shape = [1] + embedding_feature_shape[1:]
    mask = tf.ones(shape=mask_shape)
    for m in masks:
      mask = mask * tf.cast(m, 'float32')
    mask = tf.expand_dims(mask, -1)
    embedding = embedding * mask

    if self._feature_group.encoder_type == input_config_pb2.EncoderType.BOW:
      weighted_summed_context_embedding = tf.keras.backend.sum(
          embedding, axis=1)
      total_weights = tf.ones_like(
          weighted_summed_context_embedding) * tf.keras.backend.sum(
              mask, axis=1)
      embedding = safe_div(weighted_summed_context_embedding, total_weights)
    elif self._feature_group.encoder_type == input_config_pb2.EncoderType.CNN:
      for conv1d_layer in self._conv1d_layers:
        embedding = conv1d_layer(embedding)
      embedding = tf.math.reduce_max(embedding, axis=1)
    elif self._feature_group.encoder_type == input_config_pb2.EncoderType.LSTM:
      embedding = self._lstm_layer(embedding)
    else:
      raise ValueError('Unsupported encoder type %s.' %
                       self._feature_group.encoder_type)

    return embedding
