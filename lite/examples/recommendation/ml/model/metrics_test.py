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
"""Tests for the keras_metrics."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from model import metrics


class KerasMetricsTest(tf.test.TestCase):

  def test_batch_recall_and_mean_rank(self):
    batch_recall = metrics.BatchRecall(top_k=2)
    batch_mean_rank = metrics.BatchMeanRank()
    true_label = tf.constant([[2], [0], [1]], dtype=tf.int32)
    logits = tf.constant([
        [0.8, 0.1, 1.1, 0.3],
        [0.2, 0.7, 0.1, 0.5],
        [0.7, 0.4, 0.9, 0.2]
    ], dtype=tf.float32)
    batch_recall.update_state(y_true=true_label, y_pred=logits)
    batch_mean_rank.update_state(y_true=true_label, y_pred=logits)
    self.assertBetween(batch_recall.result().numpy(), 0.6, 0.7)
    self.assertEqual(batch_mean_rank.result().numpy(), 1.0)

  def test_global_recall_and_mean_rank(self):
    global_recall = metrics.GlobalRecall(top_k=2)
    global_mean_rank = metrics.GlobalMeanRank()
    true_label = tf.constant([[2], [0], [1]], dtype=tf.int32)
    logits = tf.constant([
        [0.8, 0.1, 1.1, 0.3],
        [0.2, 0.7, 0.1, 0.5],
        [0.7, 0.4, 0.9, 0.2]
    ], dtype=tf.float32)
    global_recall.update_state(y_true=true_label, y_pred=logits)
    global_mean_rank.update_state(y_true=true_label, y_pred=logits)
    self.assertBetween(global_recall.result().numpy(), 0.3, 0.4)
    self.assertBetween(global_mean_rank.result().numpy(), 1.3, 1.4)


if __name__ == '__main__':
  tf.test.main()
