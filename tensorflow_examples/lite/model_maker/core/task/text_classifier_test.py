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

import filecmp
import os

import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_examples.lite.model_maker.core import compat
from tensorflow_examples.lite.model_maker.core import test_util
from tensorflow_examples.lite.model_maker.core.data_util import text_dataloader
from tensorflow_examples.lite.model_maker.core.export_format import ExportFormat
from tensorflow_examples.lite.model_maker.core.task import configs
from tensorflow_examples.lite.model_maker.core.task import model_spec as ms
from tensorflow_examples.lite.model_maker.core.task import text_classifier


class TextClassifierTest(tf.test.TestCase):
  TEST_LABELS_AND_TEXT = (('pos', 'super good'), ('neg', 'really bad.'))

  def _gen_text_dir(self, text_per_class=1):
    text_dir = os.path.join(self.get_temp_dir(), 'random_text_dir')
    if os.path.exists(text_dir):
      return text_dir
    os.mkdir(text_dir)

    for class_name, text in self.TEST_LABELS_AND_TEXT:
      class_subdir = os.path.join(text_dir, class_name)
      os.mkdir(class_subdir)
      for i in range(text_per_class):
        with open(os.path.join(class_subdir, '%d.txt' % i), 'w') as f:
          f.write(text)
    return text_dir

  def setUp(self):
    super(TextClassifierTest, self).setUp()
    self.text_dir = self._gen_text_dir()
    self.tiny_text_dir = self._gen_text_dir(text_per_class=1)

  @test_util.test_in_tf_1
  def test_average_wordvec_model_create_v1_incompatible(self):
    with self.assertRaisesRegex(ValueError, 'Incompatible versions'):
      model_spec = ms.AverageWordVecModelSpec(seq_len=2)
      all_data = text_dataloader.TextClassifierDataLoader.from_folder(
          self.text_dir, model_spec=model_spec)
      _ = text_classifier.create(
          all_data,
          model_spec=model_spec,
      )

  @test_util.test_in_tf_2
  def test_bert_model(self):
    model_spec = ms.BertClassifierModelSpec(seq_len=2, trainable=False)
    all_data = text_dataloader.TextClassifierDataLoader.from_folder(
        self.tiny_text_dir, model_spec=model_spec)
    # Splits data, 50% data for training, 50% for testing
    self.train_data, self.test_data = all_data.split(0.5)

    model = text_classifier.create(
        self.train_data,
        model_spec=model_spec,
        epochs=1,
        batch_size=1,
        shuffle=True)
    self._test_accuracy(model, 0.0)
    self._test_export_to_tflite(
        model,
        threshold=0.0,
        expected_json_file='bert_classifier_metadata.json')
    self._test_model_without_training(model_spec)

  @test_util.test_in_tf_2
  def test_mobilebert_model(self):
    model_spec = ms.mobilebert_classifier_spec
    model_spec.seq_len = 2
    model_spec.trainable = False
    all_data = text_dataloader.TextClassifierDataLoader.from_folder(
        self.tiny_text_dir, model_spec=model_spec)
    # Splits data, 50% data for training, 50% for testing
    self.train_data, self.test_data = all_data.split(0.5)

    model = text_classifier.create(
        self.train_data,
        model_spec=model_spec,
        epochs=1,
        batch_size=1,
        shuffle=True)
    self._test_accuracy(model, 0.0)
    self._test_export_to_tflite(model, threshold=0.0, atol=1e-2)
    self._test_export_to_tflite_quant(model)

  @test_util.test_in_tf_2
  def test_average_wordvec_model(self):
    model_spec = ms.AverageWordVecModelSpec(seq_len=2)
    all_data = text_dataloader.TextClassifierDataLoader.from_folder(
        self.text_dir, model_spec=model_spec)
    # Splits data, 90% data for training, 10% for testing
    self.train_data, self.test_data = all_data.split(0.5)

    model = text_classifier.create(
        self.train_data,
        model_spec=model_spec,
        epochs=1,
        batch_size=1,
        shuffle=True)
    self._test_accuracy(model, threshold=0.0)
    self._test_predict_top_k(model)
    self._test_export_to_tflite(
        model,
        threshold=0.0,
        expected_json_file='average_word_vec_metadata.json')
    self._test_export_to_saved_model(model)
    self._test_export_labels(model)
    self._test_export_vocab(model)
    self._test_model_without_training(model_spec)

  def _test_model_without_training(self, model_spec):
    # Test without retraining.
    model = text_classifier.create(
        self.train_data, model_spec=model_spec, do_train=False)
    self._test_accuracy(model, threshold=0.0)
    self._test_export_to_tflite(model, threshold=0.0)

  def _test_accuracy(self, model, threshold=1.0):
    _, accuracy = model.evaluate(self.test_data)
    self.assertGreaterEqual(accuracy, threshold)

  def _test_predict_top_k(self, model):
    topk = model.predict_top_k(self.test_data, batch_size=1)
    for i in range(self.test_data.size):
      predict_label, predict_prob = topk[i][0][0], topk[i][0][1]
      self.assertIn(predict_label, model.index_to_label)
      self.assertGreater(predict_prob, 0.5)

  def _load_vocab(self, filepath):
    with tf.io.gfile.GFile(filepath, 'r') as f:
      return [vocab.strip('\n').split() for vocab in f]

  def _load_labels(self, filepath):
    with tf.io.gfile.GFile(filepath, 'r') as f:
      return [label.strip('\n') for label in f]

  def _test_export_labels(self, model):
    labels_output_file = os.path.join(self.get_temp_dir(), 'labels.txt')
    model.export(self.get_temp_dir(), export_format=ExportFormat.LABEL)

    labels = self._load_labels(labels_output_file)
    self.assertEqual(labels, ['neg', 'pos'])

  def _test_export_vocab(self, model):
    vocab_output_file = os.path.join(self.get_temp_dir(), 'vocab')
    model.export(self.get_temp_dir(), export_format=ExportFormat.VOCAB)

    word_index = self._load_vocab(vocab_output_file)
    expected_predefined = [['<PAD>', '0'], ['<START>', '1'], ['<UNKNOWN>', '2']]
    self.assertEqual(word_index[:3], expected_predefined)

    expected_vocab = ['bad', 'good', 'really', 'super']
    actual_vocab = sorted([word for word, index in word_index[3:]])
    self.assertEqual(actual_vocab, expected_vocab)

    expected_index = ['3', '4', '5', '6']
    actual_index = [index for word, index in word_index[3:]]
    self.assertEqual(actual_index, expected_index)

  def _test_export_to_tflite(self,
                             model,
                             threshold=1.0,
                             atol=1e-04,
                             expected_json_file=None):
    tflite_output_file = os.path.join(self.get_temp_dir(), 'model.tflite')
    model.export(self.get_temp_dir(), export_format=ExportFormat.TFLITE)

    self.assertTrue(tf.io.gfile.exists(tflite_output_file))
    self.assertGreater(os.path.getsize(tflite_output_file), 0)

    result = model.evaluate_tflite(tflite_output_file, self.test_data)
    self.assertGreaterEqual(result['accuracy'], threshold)

    spec = model.model_spec
    if isinstance(spec, ms.AverageWordVecModelSpec):
      random_inputs = np.random.randint(
          low=0, high=len(spec.vocab), size=(1, spec.seq_len), dtype=np.int32)
    elif isinstance(spec, ms.BertClassifierModelSpec):
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
    else:
      raise ValueError('Unsupported model_spec type: %s' % str(type(spec)))

    self.assertTrue(
        test_util.is_same_output(
            tflite_output_file, model.model, random_inputs, spec, atol=atol))

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

  def _test_export_to_tflite_quant(self, model):
    tflite_filename = 'model_quant.tflite'
    tflite_output_file = os.path.join(self.get_temp_dir(), tflite_filename)
    config = configs.QuantizationConfig.create_dynamic_range_quantization(
        optimizations=[tf.lite.Optimize.OPTIMIZE_FOR_LATENCY])
    model.export(
        self.get_temp_dir(),
        tflite_filename=tflite_filename,
        export_format=ExportFormat.TFLITE,
        quantization_config=config)

    self.assertTrue(tf.io.gfile.exists(tflite_output_file))
    self.assertGreater(os.path.getsize(tflite_output_file), 0)


if __name__ == '__main__':
  compat.setup_tf_behavior(tf_version=2)
  tf.test.main()
