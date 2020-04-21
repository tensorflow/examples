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

import tensorflow.compat.v2 as tf
from tensorflow_examples.lite.model_maker.core import model_export_format as mef
from tensorflow_examples.lite.model_maker.core import test_util
from tensorflow_examples.lite.model_maker.core.task import classification_model


class MockClassificationModel(classification_model.ClassificationModel):

  def train(self, train_data, validation_data=None, **kwargs):
    pass

  def export(self, **kwargs):
    pass

  def evaluate(self, data, **kwargs):
    pass


class ClassificationModelTest(tf.test.TestCase):

  def test_predict_top_k(self):
    input_shape = [24, 24, 3]
    num_classes = 2
    model = MockClassificationModel(
        model_export_format=mef.ModelExportFormat.TFLITE,
        model_spec=None,
        index_to_label=['pos', 'neg'],
        num_classes=2,
        train_whole_model=False,
        shuffle=False)
    model.model = test_util.build_model(input_shape, num_classes)
    data = test_util.get_dataloader(2, input_shape, num_classes)

    topk_results = model.predict_top_k(data, k=2, batch_size=1)
    for topk_result in topk_results:
      top1_result, top2_result = topk_result[0], topk_result[1]
      top1_label, top1_prob = top1_result[0], top1_result[1]
      top2_label, top2_prob = top2_result[0], top2_result[1]

      self.assertIn(top1_label, model.index_to_label)
      self.assertIn(top2_label, model.index_to_label)
      self.assertNotEqual(top1_label, top2_label)

      self.assertLessEqual(top1_prob, 1)
      self.assertGreaterEqual(top1_prob, top2_prob)
      self.assertGreaterEqual(top2_prob, 0)

      self.assertEqual(top1_prob + top2_prob, 1.0)


if __name__ == '__main__':
  tf.test.main()
