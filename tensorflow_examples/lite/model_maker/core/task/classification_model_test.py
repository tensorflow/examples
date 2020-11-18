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

import tensorflow.compat.v2 as tf
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

  def setUp(self):
    super(ClassificationModelTest, self).setUp()
    self.num_classes = 2
    self.model = MockClassificationModel(
        model_spec=None,
        index_to_label=['pos', 'neg'],
        train_whole_model=False,
        shuffle=False)

  def test_predict_top_k(self):
    input_shape = [24, 24, 3]
    self.model.model = test_util.build_model(input_shape, self.num_classes)
    data = test_util.get_dataloader(2, input_shape, self.num_classes)

    topk_results = self.model.predict_top_k(data, k=2, batch_size=1)
    for topk_result in topk_results:
      top1_result, top2_result = topk_result[0], topk_result[1]
      top1_label, top1_prob = top1_result[0], top1_result[1]
      top2_label, top2_prob = top2_result[0], top2_result[1]

      self.assertIn(top1_label, self.model.index_to_label)
      self.assertIn(top2_label, self.model.index_to_label)
      self.assertNotEqual(top1_label, top2_label)

      self.assertLessEqual(top1_prob, 1)
      self.assertGreaterEqual(top1_prob, top2_prob)
      self.assertGreaterEqual(top2_prob, 0)

      self.assertEqual(top1_prob + top2_prob, 1.0)

  def test_export_labels(self):
    labels_output_file = os.path.join(self.get_temp_dir(), 'label')
    self.model._export_labels(labels_output_file)
    with tf.io.gfile.GFile(labels_output_file, 'r') as f:
      labels = [label.strip() for label in f]
    self.assertEqual(labels, ['pos', 'neg'])


if __name__ == '__main__':
  tf.test.main()
