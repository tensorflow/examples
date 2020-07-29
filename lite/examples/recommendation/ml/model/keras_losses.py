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
"""Customized losses."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class BatchSoftmax(tf.keras.losses.Loss):
  """Compute batch softmax over similarities between context/label embeddings.

  Since model output is pre-calulated similarities matrix, y_pred is expected
  to be the similarities matrix.
  """

  def __init__(self, name='batch_softmax', **kwargs):
    super(BatchSoftmax, self).__init__(name=name, **kwargs)

  @tf.function
  def call(self, y_true: tf.Tensor, y_pred: tf.Tensor):
    del y_true
    logits = tf.keras.backend.cast(y_pred, 'float32')
    # Use the diagonal elements of similarities as labels for each row.
    full_labels = tf.eye(
        tf.shape(logits)[0], tf.shape(logits)[1], dtype=tf.dtypes.float32)
    batch_loss = tf.nn.softmax_cross_entropy_with_logits(
        labels=full_labels, logits=logits)
    loss = tf.reduce_mean(batch_loss)
    tf.summary.scalar('loss', loss)
    return loss
