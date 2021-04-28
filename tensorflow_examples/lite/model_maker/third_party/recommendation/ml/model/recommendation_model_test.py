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
"""Tests for personalized recommendation model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_examples.lite.model_maker.third_party.recommendation.ml.model import recommendation_model


class RecommendationModelTest(tf.test.TestCase):

  def test_model_train(self):
    config = {
        "context_embedding_dim": 128,
        "label_embedding_dim": 32,
        "hidden_layer_dim_ratios": [1, 0.5, 0.25],
        "item_vocab_size": 16,
        "encoder_type": "bow"
    }
    batch_size = 128
    test_model = recommendation_model.RecommendationModel(config)
    input_context = tf.keras.layers.Input(
        shape=(None,), dtype=tf.int32, batch_size=batch_size, name="context")
    input_label = tf.keras.layers.Input(
        shape=(1,), dtype=tf.int32, batch_size=batch_size, name="label")
    inputs = {"context": input_context, "label": input_label}
    logits = test_model(inputs)
    self.assertAllEqual([batch_size, config["item_vocab_size"] + 1],
                        logits.shape.as_list())

  def test_model_serve(self):
    config = {
        "context_embedding_dim": 128,
        "label_embedding_dim": 32,
        "hidden_layer_dim_ratios": [1, 0.5, 0.25],
        "item_vocab_size": 16,
        "encoder_type": "bow",
        "num_predictions": 10
    }
    test_model = recommendation_model.RecommendationModel(config)
    input_context = tf.constant([1, 2, 3, 4, 5])
    outputs = test_model.serve(input_context)
    self.assertAllEqual([10], outputs["top_prediction_ids"].shape.as_list())
    self.assertAllEqual([10], outputs["top_prediction_scores"].shape.as_list())


if __name__ == "__main__":
  tf.test.main()
