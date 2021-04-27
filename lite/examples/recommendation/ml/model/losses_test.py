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
"""Tests for the keras losses."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_examples.lite.examples.recommendation.ml.model import keras_losses


class KerasLossesTest(tf.test.TestCase):

  def test_batch_softmax_loss(self):
    batch_softmax = keras_losses.BatchSoftmax()
    true_label = tf.constant([[2], [0], [1]], dtype=tf.int32)
    logits = tf.constant([
        [0.8, 0.1, 0.2, 0.3],
        [0.2, 0.7, 0.1, 0.5],
        [0.5, 0.4, 0.9, 0.2]
    ], dtype=tf.float32)
    self.assertBetween(
        batch_softmax.call(y_true=true_label, y_pred=logits).numpy(),
        1.3, 1.4)

  def test_global_softmax_loss(self):
    global_softmax = keras_losses.GlobalSoftmax()
    true_label = tf.constant([[2], [0], [1]], dtype=tf.int32)
    logits = tf.constant([
        [0.8, 0.1, 0.2, 0.3],
        [0.2, 0.7, 0.1, 0.5],
        [0.5, 0.4, 0.9, 0.2]
    ], dtype=tf.float32)
    self.assertBetween(
        global_softmax.call(y_true=true_label, y_pred=logits).numpy(), 1.5, 1.6)


if __name__ == '__main__':
  tf.test.main()
