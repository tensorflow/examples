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
"""On-device personalized recommendation model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf
from model import keras_layers as layers


class RecommendationModel(tf.keras.Model):
  """Personalized dual-encoder style recommendation model."""

  def __init__(self, params):
    super(RecommendationModel, self).__init__()
    self._params = params
    self._encoder_type = self._params['encoder_type']
    # Check encoder type is valid
    self._context_embedding_dim = self._params['context_embedding_dim']
    self._label_embedding_dim = self._params['label_embedding_dim']
    self._item_vocab_size = self._params['item_vocab_size']
    # Use id=0 as padding id, so embedding layer vocab size need to add 1.
    embedding_vocab_size = self._item_vocab_size + 1

    # Prepares context embedding layer.
    self._context_embedding_layer = tf.keras.layers.Embedding(
        embedding_vocab_size,
        self._context_embedding_dim,
        embeddings_initializer=tf.keras.initializers.TruncatedNormal(
            mean=0.0, stddev=1.0 / math.sqrt(self._context_embedding_dim)),
        mask_zero=True,
        name='item_embedding_context')
    # Prepares label embedding layer.
    self._label_embedding_layer = tf.keras.layers.Embedding(
        embedding_vocab_size,
        self._label_embedding_dim,
        embeddings_initializer=tf.keras.initializers.TruncatedNormal(
            mean=0.0, stddev=1.0 / math.sqrt(self._label_embedding_dim)),
        mask_zero=True,
        name='item_embedding_label')
    # Prepare context encoder layer.
    self._context_encoder = layers.ContextEncoder(
        embedding_layer=self._context_embedding_layer,
        encoder_type=self._encoder_type,
        params=self._params)
    # Prepares the context/label embeddings dotproduct similarity layer.
    self._dotproduct_layer = layers.DotProductSimilarity()

  def _get_dotproduct_and_top_items(self,
                                    input_context,
                                    input_label,
                                    top_k=None):
    if input_context.shape.ndims == 1:
      input_context = tf.expand_dims(input_context, 0)
    context_embeddings = self._context_encoder(input_context)
    label_embeddings = self._label_embedding_layer(input_label)
    return self._dotproduct_layer(
        context_embeddings=context_embeddings,
        label_embeddings=label_embeddings,
        top_k=top_k)

  def call(self, inputs):
    """Compute outputs by passing inputs through the model.

    Here full vocab labels are used to produce dotproduct with context, all
    non-label items in the vocab will be used as negatives.

    Args:
      inputs: The inputs to the model, which should be a dictionary having
        'context' and 'label' as keys. If it's not training mode, only 'context'
        input tensor is needed.

    Returns:
      dotproduct similarity for training mode, top k prediction ids and scores
      for inference mode.
    """
    # Compute the similarities between the context embedding and embeddings of
    # all items in the vocabulary. Since the label embedding layer needs
    # to take care of out-of-vocab ID 0, the size of it is item_vocab_size + 1.
    full_vocab_item_ids = tf.range(self._item_vocab_size + 1)
    dotproduct = self._get_dotproduct_and_top_items(
        input_context=inputs['context'], input_label=full_vocab_item_ids)[0]
    return dotproduct

  @tf.function
  def serve(self, input_context):
    assert self._params['num_predictions']
    (_, top_ids, top_scores) = self._get_dotproduct_and_top_items(
        input_context=input_context,
        input_label=tf.range(self._item_vocab_size + 1),
        top_k=self._params['num_predictions'])
    return {'top_prediction_ids': top_ids, 'top_prediction_scores': top_scores}
