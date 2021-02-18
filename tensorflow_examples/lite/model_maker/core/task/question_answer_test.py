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

import filecmp
import os

from absl.testing import parameterized
import numpy as np
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

  @test_util.test_in_tf_2
  def test_bert_model(self):
    # Only test squad1.1 since it takes too long time for this.
    version = '1.1'
    model_spec = ms.BertQAModelSpec(trainable=False, predict_batch_size=1)
    train_data, validation_data = _get_data(model_spec, version)
    model = question_answer.create(
        train_data, model_spec=model_spec, epochs=1, batch_size=1)
    self._test_f1_score(model, validation_data, 0.0)
    self._test_export_vocab(model)
    self._test_export_to_tflite(model, validation_data)
    self._test_export_to_saved_model(model)
    # Comments this due to Out of Memory Error.
    # self._test_model_without_training(model_spec, train_data, validation_data)

  def _test_model_without_training(self, model_spec, train_data,
                                   validation_data):
    # Test without retraining.
    model = question_answer.create(
        train_data, model_spec=model_spec, do_train=False)
    self._test_f1_score(model, validation_data, 0.0)
    self._test_export_to_tflite(model, validation_data)

  @parameterized.parameters(
      ('mobilebert_qa', False),
      ('mobilebert_qa_squad', True),
  )
  @test_util.test_in_tf_2
  def test_mobilebert_model(self, spec, trainable):
    # Only test squad1.1 since it takes too long time for this.
    version = '1.1'
    model_spec = ms.get(spec)
    model_spec.trainable = trainable
    model_spec.predict_batch_size = 1
    train_data, validation_data = _get_data(model_spec, version)
    model = question_answer.create(
        train_data, model_spec=model_spec, epochs=1, batch_size=1)
    self._test_f1_score(model, validation_data, 0.0)
    self._test_export_to_tflite(model, validation_data, atol=1e-02)

  def _test_f1_score(self, model, validation_data, threshold):
    metric = model.evaluate(validation_data)
    self.assertGreaterEqual(metric['final_f1'], threshold)

  def _test_export_vocab(self, model):
    vocab_output_file = os.path.join(self.get_temp_dir(), 'vocab')
    model.export(self.get_temp_dir(), export_format=ExportFormat.VOCAB)

    self.assertTrue(os.path.isfile(vocab_output_file))
    self.assertGreater(os.path.getsize(vocab_output_file), 0)

  def _test_export_to_tflite(self,
                             model,
                             validation_data,
                             threshold=0.0,
                             atol=1e-04,
                             expected_json_file=None):
    tflite_output_file = os.path.join(self.get_temp_dir(), 'model.tflite')
    model.export(self.get_temp_dir(), export_format=ExportFormat.TFLITE)

    self.assertTrue(os.path.isfile(tflite_output_file))
    self.assertGreater(os.path.getsize(tflite_output_file), 0)

    metric = model.evaluate_tflite(tflite_output_file, validation_data)
    self.assertGreaterEqual(metric['final_f1'], threshold)

    spec = model.model_spec
    input_word_ids = np.random.randint(
        low=0,
        high=len(spec.tokenizer.vocab),
        size=(1, spec.seq_len),
        dtype=np.int32)
    input_mask = np.random.randint(
        low=0, high=2, size=(1, spec.seq_len), dtype=np.int32)
    input_type_ids = np.random.randint(
        low=0, high=2, size=(1, spec.seq_len), dtype=np.int32)
    random_inputs = (input_word_ids, input_mask, input_type_ids)

    self.assertTrue(
        test_util.is_same_output(
            tflite_output_file,
            model.model,
            random_inputs,
            model.model_spec,
            atol=atol))

    json_output_file = os.path.join(self.get_temp_dir(), 'model.json')
    self.assertTrue(os.path.isfile(json_output_file))
    self.assertGreater(os.path.getsize(json_output_file), 0)

    if expected_json_file is not None:
      expected_json_file = test_util.get_test_data_path(expected_json_file)
      self.assertTrue(filecmp.cmp(json_output_file, expected_json_file))

  def _test_export_to_saved_model(self, model):
    save_model_output_path = os.path.join(self.get_temp_dir(), 'saved_model')
    model.export(self.get_temp_dir(), export_format=ExportFormat.SAVED_MODEL)

    self.assertTrue(os.path.isdir(save_model_output_path))
    self.assertNotEmpty(os.listdir(save_model_output_path))


if __name__ == '__main__':
  # Load compressed models from tensorflow_hub
  os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'
  compat.setup_tf_behavior(tf_version=2)
  tf.test.main()
