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

from tensorflow_examples.lite.model_maker.core.data_util import recommendation_testutil as _testutil
from tensorflow_examples.lite.model_maker.core.task.model_spec import recommendation_spec
from tensorflow_examples.lite.model_maker.third_party.recommendation.ml.model import recommendation_model as _model


class RecommendationSpecTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(
      ('bow'),
      ('cnn'),
      ('lstm'),
  )
  def test_create_recommendation_model(self, encoder_type):
    input_spec = _testutil.get_input_spec(encoder_type)
    model_hparams = _testutil.get_model_hparams()
    spec = recommendation_spec.RecommendationSpec(input_spec, model_hparams)
    model = spec.create_model()
    self.assertIsInstance(model, _model.RecommendationModel)


if __name__ == '__main__':
  tf.test.main()
