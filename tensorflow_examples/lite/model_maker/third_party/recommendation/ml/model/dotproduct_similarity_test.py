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
"""Tests for dotproduct_similarity."""
import tensorflow as tf

from tensorflow_examples.lite.model_maker.third_party.recommendation.ml.model import dotproduct_similarity


class DotproductSimilarityTest(tf.test.TestCase):

  def test_dotproduct(self):
    context_embeddings = tf.constant([[0.1, 0.1, 0.1, 0.1],
                                      [0.2, 0.2, 0.2, 0.2]])
    label_embeddings = tf.constant([[0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]])
    similarity_layer = dotproduct_similarity.DotProductSimilarity()
    dotproduct = similarity_layer(context_embeddings, label_embeddings)
    dotproduct, top_ids, top_scores = similarity_layer(context_embeddings,
                                                       label_embeddings, 1)
    self.assertAllClose(tf.constant([[0.04, 0.04], [0.08, 0.08]]), dotproduct)
    self.assertAllEqual(tf.constant([[0], [0]]), top_ids)
    self.assertAllEqual([2, 1], list(top_scores.shape))

if __name__ == '__main__':
  tf.test.main()
