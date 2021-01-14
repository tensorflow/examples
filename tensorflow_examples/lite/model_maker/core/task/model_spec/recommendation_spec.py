# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Recommendation model specification."""

import functools

import tensorflow as tf  # pylint: disable=unused-import

HAS_RECOMMENDATION = True
try:
  from tensorflow_examples.lite.model_maker.third_party.recommendation.ml.model import recommendation_model as rm  # pylint: disable=g-import-not-at-top
except ImportError:
  HAS_RECOMMENDATION = False


class RecommendationSpec(object):
  """Recommendation model spec."""

  def __init__(self,
               encoder_type='bow',
               context_embedding_dim=128,
               label_embedding_dim=32,
               item_vocab_size=16,
               num_predictions=10,
               hidden_layer_dim_ratios=None,
               conv_num_filter_ratios=None,
               conv_kernel_size=None,
               lstm_num_units=None):
    """Initialize spec.

    Args:
      encoder_type: str, encoder type. One of ('bow', 'cnn', 'rnn').
      context_embedding_dim: int, dimension of context embedding layer.
      label_embedding_dim: int, dimension of label embedding layer.
      item_vocab_size: int, the size of items to be predict.
      num_predictions: int, the number of top-K predictions in the output.
      hidden_layer_dim_ratios: list of float, number of units in hidden layers
        specified by ratios. default: [1.0, 0.5, 0.25].
      conv_num_filter_ratios: list of int, for 'cnn', Conv1D layers' filter
        ratios based on context_embedding_dim.
      conv_kernel_size: int, for 'rnn', Conv1D layers' kernel size.
      lstm_num_units: int, for 'rnn', LSTM layer's unit number.
    """
    hidden_layer_dim_ratios = hidden_layer_dim_ratios or [1.0, 0.5, 0.25]

    if encoder_type == 'cnn':
      conv_num_filter_ratios = conv_num_filter_ratios or [2, 4]
      conv_kernel_size = conv_kernel_size or 4
    elif encoder_type == 'rnn':
      lstm_num_units = lstm_num_units or 16

    self.encoder_type = encoder_type
    self.context_embedding_dim = context_embedding_dim
    self.label_embedding_dim = label_embedding_dim
    self.hidden_layer_dim_ratios = hidden_layer_dim_ratios
    self.item_vocab_size = item_vocab_size
    self.num_predictions = num_predictions
    self.conv_num_filter_ratios = conv_num_filter_ratios
    self.conv_kernel_size = conv_kernel_size
    self.lstm_num_units = lstm_num_units

    self._params = {
        'encoder_type': encoder_type,
        'context_embedding_dim': context_embedding_dim,
        'label_embedding_dim': label_embedding_dim,
        'hidden_layer_dim_ratios': hidden_layer_dim_ratios,
        'item_vocab_size': item_vocab_size,
        'num_predictions': num_predictions,
        'conv_num_filter_ratios': conv_num_filter_ratios,
        'conv_kernel_size': conv_kernel_size,
        'lstm_num_units': lstm_num_units,
    }

  def create_model(self):
    """Creates recommendation model based on params."""
    if not HAS_RECOMMENDATION:
      return None
    return rm.RecommendationModel(self._params)


recommendation_bow_spec = functools.partial(
    RecommendationSpec, encoder_type='bow')
recommendation_cnn_spec = functools.partial(
    RecommendationSpec, encoder_type='cnn')
recommendation_rnn_spec = functools.partial(
    RecommendationSpec, encoder_type='rnn')
