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
"""Dotproduct similarity layer."""
import tensorflow as tf


class DotProductSimilarity(tf.keras.layers.Layer):
  """Layer to comput dotproduct similarities for context/label embedding.

    The top_k is an integer to represent top_k ids to compute among label ids.
    if top_k is None, top_k computation will be ignored.
  """

  def call(self, context_embeddings: tf.Tensor, label_embeddings: tf.Tensor,
           top_k: int):
    """Generate dotproduct similarity matrix and top values/indices.

    Args:
      context_embeddings: Context embeddings generated with input context
        sequence. The shape of tensor should be [batch_size, embedding_dim] for
        training mode, and [1, embedding_dim] for inference mode.
      label_embeddings: Label embeddings generated for candidate label items.
        The shape of tensor should be [num_items, embedding_dim].
      top_k: The number of top values to compute. If it's not set, top values
        and indices will not be computed.

    Returns:
      Dotproduct similarity matrix with shape [num_label_items, 1] and top
      values/indices if top_k is set.
    """
    if tf.keras.backend.ndim(label_embeddings) == 3:
      label_embeddings = tf.squeeze(label_embeddings,
                                    1)
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
