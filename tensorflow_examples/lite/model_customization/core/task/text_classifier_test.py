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
import numpy as np
import tensorflow as tf # TF2

from tensorflow_examples.lite.model_customization.core import compat
from tensorflow_examples.lite.model_customization.core.data_util import text_dataloader
import tensorflow_examples.lite.model_customization.core.model_export_format as mef
from tensorflow_examples.lite.model_customization.core.task import text_classifier
import tensorflow_examples.lite.model_customization.core.task.model_spec as ms


class TextClassifierTest(tf.test.TestCase):
  TEXT_PER_CLASS = 20
  TEST_LABELS_AND_TEXT = (('pos', 'super good'), ('neg', 'really bad.'))

  def _gen(self):
    for i, (_, text) in enumerate(self.TEST_LABELS_AND_TEXT):
      for _ in range(self.TEXT_PER_CLASS):
        yield text, i

  def _gen_text_dir(self):
    text_dir = os.path.join(self.get_temp_dir(), 'random_text_dir')
    if os.path.exists(text_dir):
      return text_dir
    os.mkdir(text_dir)

    for class_name, text in self.TEST_LABELS_AND_TEXT:
      class_subdir = os.path.join(text_dir, class_name)
      os.mkdir(class_subdir)
      for i in range(self.TEXT_PER_CLASS):
        with open(os.path.join(class_subdir, '%d.txt' % i), 'w') as f:
          f.write(text)
    return text_dir

  def setUp(self):
    super(TextClassifierTest, self).setUp()
    self.text_dir = self._gen_text_dir()

  @compat.test_in_tf_1
  def test_average_wordvec_model_create_v1_incompatible(self):
    with self.assertRaisesRegex(ValueError, 'Incompatible versions'):
      model_spec = ms.AverageWordVecModelSpec(seq_len=2)
      all_data = text_dataloader.TextClassifierDataLoader.from_folder(
          self.text_dir, model_spec=model_spec)
      _ = text_classifier.create(
          all_data,
          mef.ModelExportFormat.TFLITE,
          model_spec=model_spec,
      )

  @compat.test_in_tf_2
  def test_average_wordvec_model(self):
    model_spec = ms.AverageWordVecModelSpec(seq_len=2)
    all_data = text_dataloader.TextClassifierDataLoader.from_folder(
        self.text_dir, model_spec=model_spec)
    # Splits data, 90% data for training, 10% for testing
    self.train_data, self.test_data = all_data.split(0.9)

    model = text_classifier.create(
        self.train_data,
        mef.ModelExportFormat.TFLITE,
        model_spec=model_spec,
        epochs=2,
        batch_size=4,
        shuffle=True)
    self._test_accuracy(model)
    self._test_export_to_tflite(model)
    self._test_predict_top_k(model)

  def _test_accuracy(self, model):
    _, accuracy = model.evaluate(self.test_data)
    self.assertEqual(accuracy, 1.0)

  def _test_predict_top_k(self, model):
    topk = model.predict_top_k(self.test_data, batch_size=4)
    for i, (_, label) in enumerate(self.test_data.dataset):
      predict_label, predict_prob = topk[i][0][0], topk[i][0][1]
      self.assertEqual(model.index_to_label[label], predict_label)
      self.assertGreater(predict_prob, 0.5)

  def _load_vocab(self, filename):
    with tf.io.gfile.GFile(filename, 'r') as f:
      return [vocab.strip('\n').split() for vocab in f]

  def _load_labels(self, filename):
    with tf.io.gfile.GFile(filename, 'r') as f:
      return [label.strip('\n') for label in f]

  def _load_lite_model(self, filename):
    self.assertTrue(os.path.isfile(filename))
    with tf.io.gfile.GFile(filename, 'rb') as f:
      model_content = f.read()
    interpreter = tf.lite.Interpreter(model_content=model_content)

    def lite_model(text):
      interpreter.allocate_tensors()
      input_index = interpreter.get_input_details()[0]['index']
      input_shape = interpreter.get_input_details()[0]['shape']
      interpreter.set_tensor(input_index, tf.reshape(text, input_shape))
      interpreter.invoke()
      output_index = interpreter.get_output_details()[0]['index']
      return interpreter.get_tensor(output_index)

    return lite_model

  def _test_export_to_tflite(self, model):
    tflite_output_file = os.path.join(self.get_temp_dir(), 'model.tflite')
    labels_output_file = os.path.join(self.get_temp_dir(), 'label')
    vocab_output_file = os.path.join(self.get_temp_dir(), 'vocab')
    model.export(tflite_output_file, labels_output_file, vocab_output_file)

    labels = self._load_labels(labels_output_file)
    self.assertEqual(labels, ['neg', 'pos'])

    word_index = self._load_vocab(vocab_output_file)
    expected_predefined = [['<PAD>', '0'], ['<START>', '1'], ['<UNKNOWN>', '2']]
    self.assertEqual(word_index[:3], expected_predefined)

    expected_vocab = ['bad', 'good', 'really', 'super']
    actual_vocab = sorted([word for word, index in word_index[3:]])
    self.assertEqual(actual_vocab, expected_vocab)

    expected_index = ['3', '4', '5', '6']
    actual_index = [index for word, index in word_index[3:]]
    self.assertEqual(actual_index, expected_index)

    lite_model = self._load_lite_model(tflite_output_file)
    for x, y in self.test_data.dataset:
      input_batch = tf.cast(x, tf.float32)
      output_batch = lite_model(input_batch)
      prediction = np.argmax(output_batch[0])
      self.assertEqual(y, prediction)


if __name__ == '__main__':
  compat.setup_tf_behavior(tf_version=2)
  tf.test.main()
