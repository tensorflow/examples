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
"""Tests for recommendation spec."""

from absl.testing import parameterized
import tensorflow.compat.v2 as tf

from tensorflow_examples.lite.model_maker.core.task.model_spec import recommendation_spec


class RecommendationSpecTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(
      ('bow'),
      ('cnn'),
      ('rnn'),
  )
  def test_create_recommendation_model(self, encoder_type):
    spec = recommendation_spec.RecommendationSpec(encoder_type)
    model = spec.create_model()
    if recommendation_spec.HAS_RECOMMENDATION:
      self.assertIsInstance(model, recommendation_spec.rm.RecommendationModel)
    else:
      self.assertIsNone(model)


if __name__ == '__main__':
  tf.test.main()
