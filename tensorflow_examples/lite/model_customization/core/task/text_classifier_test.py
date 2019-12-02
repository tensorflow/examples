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

from tensorflow_examples.lite.model_customization.core.data_util import text_dataloader
import tensorflow_examples.lite.model_customization.core.model_export_format as mef
from tensorflow_examples.lite.model_customization.core.task import text_classifier


class TextClassifierTest(tf.test.TestCase):
  TEXT_PER_CLASS = 20
  TEST_LABELS_AND_TEXT = (('pos', 'super good'), ('neg', 'really bad.'))

  def _gen(self):
    for i, (_, text) in enumerate(self.TEST_LABELS_AND_TEXT):
      for _ in range(self.TEXT_PER_CLASS):
        yield text, i

  def _gen_data(self):
    ds = tf.data.Dataset.from_generator(
        self._gen, (tf.string, tf.int64),
        (tf.TensorShape([]), tf.TensorShape([])))
    data = text_dataloader.TextClassifierDataLoader(ds, self.TEXT_PER_CLASS * 2,
                                                    2, ['pos', 'neg'])
    return data

  def setUp(self):
    super(TextClassifierTest, self).setUp()
    self.data = self._gen_data()

  def test_average_wordvec_model(self):
    model = text_classifier.create(
        self.data,
        mef.ModelExportFormat.TFLITE,
        model_name='average_wordvec',
        epochs=2,
        batch_size=4,
        sentence_len=2,
        shuffle=True)
    self._test_accuracy(model)
    self._test_export_to_tflite(model)
    self._test_predict_topk(model)

  def _test_accuracy(self, model):
    _, accuracy = model.evaluate()
    self.assertEqual(accuracy, 1.0)

  def _test_predict_topk(self, model):
    topk = model.predict_topk(batch_size=4)
    for i, (_, label) in enumerate(model.test_data.dataset):
      predict_label, predict_prob = topk[i][0][0], topk[i][0][1]
      self.assertEqual(model.data.index_to_label[label], predict_label)
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
    self.assertEqual(labels, ['pos', 'neg'])

    word_index = self._load_vocab(vocab_output_file)
    expected = [['<PAD>', '0'], ['<START>', '1'], ['<UNKNOWN>', '2'],
                ['super', '3'], ['good', '4'], ['really', '5'], ['bad', '6']]
    self.assertEqual(word_index, expected)

    lite_model = self._load_lite_model(tflite_output_file)
    for i, (class_name, text) in enumerate(self.TEST_LABELS_AND_TEXT):
      input_batch = tf.constant(text)
      input_batch = model.preprocess_text(input_batch, np.int64(i))[0]
      input_batch = tf.cast(input_batch, tf.float32)
      output_batch = lite_model(input_batch)
      prediction = labels[np.argmax(output_batch[0])]
      self.assertEqual(class_name, prediction)


if __name__ == '__main__':
  assert tf.__version__.startswith('2')
  tf.test.main()
