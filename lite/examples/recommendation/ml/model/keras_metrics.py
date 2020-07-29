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
"""Customized metrics."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class BatchRecall(tf.keras.metrics.Mean):
  """Compute batch recall for top_k.

  Since the model output for train and eval mode is similarities between
  context/label embeddings, y_pred is expected to the similarities.
  """

  def __init__(self, name='batch_recall', top_k=1, **kwargs):
    super().__init__(name=name, **kwargs)
    self.top_k = top_k

  def update_state(self, y_true, y_pred, sample_weight=None):
    del y_true
    similarities = y_pred
    # Get indices of similar labels in sorted order for each context.
    sorted_similarities = tf.argsort(
        similarities, axis=-1, direction='DESCENDING')
    # Get the ranks of the correct label for each context.
    batch_size = tf.shape(similarities)[0]
    label_indices = tf.expand_dims(tf.range(batch_size), -1)
    ranks = tf.where(tf.equal(sorted_similarities, label_indices))[:, -1]

    # Compare the ranks with top_k to produce recall indicators for each
    # example.
    recalls = tf.keras.backend.less(ranks, self.top_k)
    super().update_state(recalls, sample_weight=sample_weight)

  def get_config(self):
    config = {'top_k': self.top_k}
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))


class BatchMeanRank(tf.keras.metrics.Mean):
  """Keras metric computing mean rank of correct label within a batch.

  Since the model output for train and eval mode is similarities between
  context/label embeddings, y_pred is expected to the similarities.
  """

  def __init__(self, name='batch_mean_rank', **kwargs):
    super().__init__(name=name, **kwargs)

  def update_state(self, y_true, y_pred, sample_weight=None):
    similarities = y_pred

    # Get the indices of similar labels in sorted order for each context.
    sorted_similarities = tf.argsort(
        similarities, axis=-1, direction='DESCENDING')

    # Get the ranks of the correct label for each context.
    batch_size = tf.shape(similarities)[0]
    label_indices = tf.expand_dims(tf.range(batch_size), -1)
    ranks = tf.where(tf.equal(sorted_similarities, label_indices))[:, -1]
    ranks = tf.keras.backend.cast(ranks, 'float32')
    super().update_state(ranks, sample_weight=sample_weight)
