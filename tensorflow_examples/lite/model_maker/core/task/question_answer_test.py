# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
import tensorflow as tf

from tensorflow_examples.lite.model_maker.core import compat
from tensorflow_examples.lite.model_maker.core import test_util
from tensorflow_examples.lite.model_maker.core.data_util import text_dataloader
from tensorflow_examples.lite.model_maker.core.export_format import ExportFormat
from tensorflow_examples.lite.model_maker.core.task import model_spec as ms
from tensorflow_examples.lite.model_maker.core.task import question_answer


def _get_data(model_spec, version):
  path = test_util.get_test_data_path('squad_testdata')
  train_data_path = os.path.join(path, 'train-v%s.json' % version)
  validation_data_path = os.path.join(path, 'dev-v%s.json' % version)
  version_2_with_negative = version.startswith('2')
  train_data = text_dataloader.QuestionAnswerDataLoader.from_squad(
      train_data_path,
      model_spec,
      is_training=True,
      version_2_with_negative=version_2_with_negative)
  validation_data = text_dataloader.QuestionAnswerDataLoader.from_squad(
      validation_data_path,
      model_spec,
      is_training=False,
      version_2_with_negative=version_2_with_negative)
  return train_data, validation_data


class QuestionAnswerTest(tf.test.TestCase, parameterized.TestCase):

  @test_util.test_in_tf_1
  def test_bert_model_v1_incompatible(self):
    with self.assertRaisesRegex(ValueError, 'Incompatible versions'):
      _ = ms.BertQAModelSpec(trainable=False)

  @parameterized.parameters(
      ('1.1'),
      ('2.0'),
  )
  @test_util.test_in_tf_2
  def test_bert_model(self, version):
    model_spec = ms.BertQAModelSpec(trainable=False, predict_batch_size=1)
    train_data, validation_data = _get_data(model_spec, version)
    model = question_answer.create(
        train_data, model_spec=model_spec, epochs=1, batch_size=1)
    self._test_f1_score(model, validation_data, 0.0)
    self._test_export_vocab(model)
    self._test_export_to_tflite(model)
    self._test_export_to_saved_model(model)

  def _test_f1_score(self, model, validation_data, threshold):
    metric = model.evaluate(validation_data)
    self.assertGreaterEqual(metric['final_f1'], threshold)

  def _test_export_vocab(self, model):
    vocab_output_file = os.path.join(self.get_temp_dir(), 'vocab')
    model.export(self.get_temp_dir(), export_format=ExportFormat.VOCAB)

    self.assertTrue(os.path.isfile(vocab_output_file))
    self.assertGreater(os.path.getsize(vocab_output_file), 0)

  def _test_export_to_tflite(self, model):
    tflite_output_file = os.path.join(self.get_temp_dir(), 'model.tflite')
    model.export(self.get_temp_dir(), export_format=ExportFormat.TFLITE)

    self.assertTrue(os.path.isfile(tflite_output_file))
    self.assertGreater(os.path.getsize(tflite_output_file), 0)

  def _test_export_to_saved_model(self, model):
    save_model_output_path = os.path.join(self.get_temp_dir(), 'saved_model')
    model.export(self.get_temp_dir(), export_format=ExportFormat.SAVED_MODEL)

    self.assertTrue(os.path.isdir(save_model_output_path))
    self.assertNotEmpty(os.listdir(save_model_output_path))


if __name__ == '__main__':
  compat.setup_tf_behavior(tf_version=2)
  tf.test.main()
