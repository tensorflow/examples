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
"""Layers for on-device personalized recommendation model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

SUPPORTED_ENCODER_TYPES = ['bow', 'rnn', 'cnn']


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

  def __init__(self, embedding_layer, encoder_type, params):
    super(ContextEncoder, self).__init__()
    self._embedding_layer = embedding_layer
    self._params = params
    self._encoder_type = encoder_type
    assert self._encoder_type in SUPPORTED_ENCODER_TYPES
    assert self._params['context_embedding_dim']
    self._context_embedding_dim = self._params['context_embedding_dim']
    # Prepare CNN and RNN layers based on encoder_type.
    self._conv1d_layers = []
    self._lstm_layer = None
    if self._encoder_type == 'cnn':
      assert self._params['conv_num_filter_ratios']
      assert self._params['conv_kernel_size']
      conv_kernel_size = self._params['conv_kernel_size']
      for ratio in self._params['conv_num_filter_ratios']:
        self._conv1d_layers.append(
            tf.keras.layers.Conv1D(
                filters=self._context_embedding_dim * ratio,
                kernel_size=conv_kernel_size,
                strides=conv_kernel_size,
                padding='same',
                activation='relu'))
    elif self._encoder_type == 'rnn':
      assert self._params['lstm_num_units']
      lstm_num_units = self._params['lstm_num_units']
      self._lstm_layer = tf.keras.layers.LSTM(lstm_num_units)
    # Prepares hidden layers.
    self._hidden_layers = []
    for i, ratio in enumerate(self._params['hidden_layer_dim_ratios']):
      self._hidden_layers.append(
          tf.keras.layers.Dense(
              units=ratio * self._context_embedding_dim,
              name='hidden_layer{}'.format(i),
              activation=tf.nn.relu))

  def call(self, input_context):
    """Encode context sequence.

    This function performs 3 steps 1) embed sequence ids 2) pass embedded
    sequence through bow/cnn/rnn neural network 3) apply full-connected hidden
    layers.

    Args:
      input_context: tensor with context sequence ids, with dimension
        [batch_size, sequence_length]

    Returns:
      context_embedding: encoded context vector, with dimension
      [batch_size, final_hidden_layer_dim].
    """
    # Embed sequence ids, masking out-of-vocabulary ids.
    context_embedding = self._embedding_layer(
        input_context)  # batch_size, sequence_length, context_embedding_dim
    mask = self._embedding_layer.compute_mask(input_context)
    mask = tf.keras.backend.expand_dims(
        tf.keras.backend.cast(mask, 'float32'), axis=-1)
    context_embedding = context_embedding * mask
    # Set up encoder by type.
    if self._encoder_type == 'bow':
      weighted_summed_context_embedding = tf.keras.backend.sum(
          context_embedding, axis=1)
      total_weights = tf.ones_like(
          weighted_summed_context_embedding) * tf.keras.backend.sum(
              mask, axis=1)
      context_embedding = safe_div(weighted_summed_context_embedding,
                                   total_weights)
    elif self._encoder_type == 'cnn':
      for conv1d_layer in self._conv1d_layers:
        context_embedding = conv1d_layer(context_embedding)
      context_embedding = tf.math.reduce_max(context_embedding, axis=1)
    elif self._encoder_type == 'rnn':
      context_embedding = self._lstm_layer(context_embedding)
    # Setup hidden layers
    for hidden_layer in self._hidden_layers:
      context_embedding = hidden_layer(context_embedding)
    return context_embedding


class DotProductSimilarity(tf.keras.layers.Layer):
  """Layer to comput dotproduct similarities for context/label embeddings.

    The top_k is an integer to represent top_k ids to compute among label ids.
    if top_k is None, top_k computation will be ignored.
  """

  def call(self, context_embeddings, label_embeddings, top_k=None, **kwargs):
    if tf.keras.backend.ndim(label_embeddings) == 3:
      label_embeddings = tf.squeeze(label_embeddings,
                                    1)  # batch_size, label_embedding_dim
    dotproduct = tf.matmul(
        context_embeddings, label_embeddings, transpose_b=True)
    if top_k:
      top_values, top_indices = tf.math.top_k(
          tf.squeeze(dotproduct), top_k, sorted=True)
      top_ids = tf.identity(top_indices, name='top_prediction_ids')
      top_scores = tf.identity(
          tf.math.sigmoid(top_values), name='top_prediction_scores')
      return [dotproduct, top_ids, top_scores]
    else:
      return [dotproduct]
