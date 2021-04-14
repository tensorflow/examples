# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl.testing import parameterized
import tensorflow.compat.v2 as tf

from tensorflow_examples.lite.model_maker.core.task import model_spec as ms

MODELS = (
    ms.IMAGE_CLASSIFICATION_MODELS + ms.TEXT_CLASSIFICATION_MODELS +
    ms.QUESTION_ANSWER_MODELS + ms.AUDIO_CLASSIFICATION_MODELS +
    ms.RECOMMENDATION_MODELS)


class ModelSpecTest(tf.test.TestCase, parameterized.TestCase):

  def test_get(self):
    spec = ms.get('mobilenet_v2')
    self.assertIsInstance(spec, ms.ImageModelSpec)

    spec = ms.get('average_word_vec')
    self.assertIsInstance(spec, ms.AverageWordVecModelSpec)

    spec = ms.get(ms.mobilenet_v2_spec)
    self.assertIsInstance(spec, ms.ImageModelSpec)

  @parameterized.parameters(MODELS)
  def test_get_not_none(self, model):
    spec = ms.get(model)
    self.assertIsNotNone(spec)

  def test_get_raises(self):
    with self.assertRaises(KeyError):
      ms.get('not_exist_model_spec')


if __name__ == '__main__':
  # Load compressed models from tensorflow_hub
  os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'
  tf.test.main()
