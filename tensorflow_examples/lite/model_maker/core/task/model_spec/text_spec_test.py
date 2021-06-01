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

import collections
import os

from absl.testing import parameterized
import tensorflow.compat.v2 as tf

from tensorflow_examples.lite.model_maker.core.task.model_spec import text_spec
from official.nlp.data import classifier_data_lib


def _gen_examples():
  examples = []
  examples.append(
      classifier_data_lib.InputExample(
          guid=0, text_a='Really good.', label='pos'))
  examples.append(
      classifier_data_lib.InputExample(guid=1, text_a='So bad.', label='neg'))
  return examples


def _get_dataset_from_tfrecord(tfrecord_file, name_to_features):

  def _parse_function(example_proto):
    # Parse the input `tf.Example` proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, name_to_features)

  ds = tf.data.TFRecordDataset(tfrecord_file)
  ds = ds.map(_parse_function)
  return ds


class AverageWordVecModelSpecTest(tf.test.TestCase):

  def setUp(self):
    super(AverageWordVecModelSpecTest, self).setUp()
    self.model_spec = text_spec.AverageWordVecModelSpec(seq_len=5)
    self.vocab = collections.OrderedDict(
        (('<PAD>', 0), ('<START>', 1), ('<UNKNOWN>', 2), ('good', 3), ('bad',
                                                                       4)))
    self.model_spec.vocab = self.vocab

  def test_tokenize(self):
    model_spec = text_spec.AverageWordVecModelSpec()
    text = model_spec._tokenize('It\'s really good.')
    self.assertEqual(text, ['it\'s', 'really', 'good'])

    model_spec = text_spec.AverageWordVecModelSpec(lowercase=False)
    text = model_spec._tokenize('That is so cool!!!')
    self.assertEqual(text, ['That', 'is', 'so', 'cool'])

  def test_convert_examples_to_features(self):
    examples = _gen_examples()
    tfrecord_file = os.path.join(self.get_temp_dir(), 'tmp.tfrecord')
    self.model_spec.convert_examples_to_features(examples, tfrecord_file,
                                                 ['pos', 'neg'])
    ds = _get_dataset_from_tfrecord(tfrecord_file,
                                    self.model_spec.get_name_to_features())

    expected_features = [[[1, 2, 3, 0, 0], 0], [[1, 2, 4, 0, 0], 1]]
    for i, sample in enumerate(ds):
      self.assertTrue(
          (sample['input_ids'].numpy() == expected_features[i][0]).all())
      self.assertEqual(sample['label_ids'].numpy(), expected_features[i][1])

  def test_preprocess(self):
    token_ids = self.model_spec.preprocess('It\'s really good.')
    expected_token_ids = [1, 2, 2, 3, 0]
    self.assertEqual(token_ids, expected_token_ids)

  def test_gen_vocab(self):
    examples = _gen_examples()
    self.model_spec.gen_vocab(examples)
    expected_vocab = collections.OrderedDict([('<PAD>', 0), ('<START>', 1),
                                              ('<UNKNOWN>', 2), ('really', 3),
                                              ('good', 4), ('so', 5),
                                              ('bad', 6)])
    self.assertEqual(self.model_spec.vocab, expected_vocab)

  def test_save_load_vocab(self):
    vocab_file = os.path.join(self.get_temp_dir(), 'vocab.txt')
    self.model_spec.save_vocab(vocab_file)
    vocab = self.model_spec.load_vocab(vocab_file)
    self.assertEqual(vocab, self.vocab)

  def test_run_classifier(self):
    num_classes = 2
    model = self.model_spec.run_classifier(
        train_ds=self._gen_random_ds(num_classes),
        validation_ds=self._gen_random_ds(num_classes),
        epochs=1,
        steps_per_epoch=1,
        num_classes=num_classes)
    self.assertIsInstance(model, tf.keras.Model)

  def _gen_random_ds(self, num_classes, data_size=1, batch_size=4):
    batched_features = tf.random.uniform(
        (data_size, batch_size, self.model_spec.seq_len),
        minval=0,
        maxval=len(self.model_spec.vocab),
        dtype=tf.dtypes.int32)

    batched_labels = tf.random.uniform((data_size, batch_size),
                                       minval=0,
                                       maxval=num_classes,
                                       dtype=tf.dtypes.int32)
    ds = tf.data.Dataset.from_tensor_slices((batched_features, batched_labels))
    return ds


class BertClassifierModelSpecTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(
      ('https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1', True),
      ('https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1', False),
  )
  def test_bert(self, uri, is_tf2):
    model_spec = text_spec.BertClassifierModelSpec(
        uri, is_tf2=is_tf2, distribution_strategy='off', seq_len=3)
    self._test_convert_examples_to_features(model_spec)
    self._test_run_classifier(model_spec)

  def _test_convert_examples_to_features(self, model_spec):
    examples = _gen_examples()
    tfrecord_file = os.path.join(self.get_temp_dir(), 'tmp.tfrecord')
    model_spec.convert_examples_to_features(examples, tfrecord_file,
                                            ['pos', 'neg'])

    ds = _get_dataset_from_tfrecord(tfrecord_file,
                                    model_spec.get_name_to_features())
    expected_features = []
    expected_features.append({
        'input_ids': [101, 2428, 102],
        'input_mask': [1, 1, 1],
        'segment_ids': [0, 0, 0],
        'label_ids': 0
    })
    expected_features.append({
        'input_ids': [101, 2061, 102],
        'input_mask': [1, 1, 1],
        'segment_ids': [0, 0, 0],
        'label_ids': 1
    })
    for i, sample in enumerate(ds):
      for k, v in expected_features[i].items():
        self.assertTrue((sample[k].numpy() == v).all())

  def _test_run_classifier(self, model_spec):
    num_classes = 2
    model = model_spec.run_classifier(
        train_ds=self._gen_random_ds(model_spec.seq_len, num_classes),
        validation_ds=self._gen_random_ds(model_spec.seq_len, num_classes),
        epochs=1,
        steps_per_epoch=1,
        num_classes=num_classes)
    self.assertIsInstance(model, tf.keras.Model)

  def _gen_random_ds(self, seq_len, num_classes, data_size=1, batch_size=1):

    batched_input_ids = tf.random.uniform((data_size, batch_size, seq_len),
                                          minval=0,
                                          maxval=2,
                                          dtype=tf.dtypes.int32)
    batched_input_mask = tf.random.uniform((data_size, batch_size, seq_len),
                                           minval=0,
                                           maxval=2,
                                           dtype=tf.dtypes.int32)
    batched_segment_ids = tf.random.uniform((data_size, batch_size, seq_len),
                                            minval=0,
                                            maxval=2,
                                            dtype=tf.dtypes.int32)

    batched_labels = tf.random.uniform((data_size, batch_size),
                                       minval=0,
                                       maxval=num_classes,
                                       dtype=tf.dtypes.int32)
    x = {
        'input_word_ids': batched_input_ids,
        'input_mask': batched_input_mask,
        'input_type_ids': batched_segment_ids
    }
    y = batched_labels

    ds = tf.data.Dataset.from_tensor_slices((x, y))
    return ds


if __name__ == '__main__':
  # Load compressed models from tensorflow_hub
  os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'
  tf.test.main()
