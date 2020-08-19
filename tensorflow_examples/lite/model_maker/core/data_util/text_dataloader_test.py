# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import json
import os

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_examples.lite.model_maker.core import test_util
from tensorflow_examples.lite.model_maker.core.data_util import text_dataloader
from official.nlp.bert import tokenization
from official.nlp.data import squad_lib


class MockClassifierModelSpec(object):
  need_gen_vocab = False

  def __init__(self, seq_len=4):
    self.seq_len = seq_len

  def get_name_to_features(self):
    """Gets the dictionary describing the features."""
    name_to_features = {
        'input_ids': tf.io.FixedLenFeature([self.seq_len], tf.int64),
        'label_ids': tf.io.FixedLenFeature([], tf.int64),
    }
    return name_to_features

  def select_data_from_record(self, record):
    """Dispatches records to features and labels."""
    x = record['input_ids']
    y = record['label_ids']
    return (x, y)

  def convert_examples_to_features(self, examples, tfrecord_file, label_names):
    """Converts examples to features and write them into TFRecord file."""
    writer = tf.io.TFRecordWriter(tfrecord_file)

    label_to_id = dict((name, i) for i, name in enumerate(label_names))
    for example in examples:
      features = collections.OrderedDict()

      label_id = label_to_id[example.label]
      input_ids = [label_id] * self.seq_len

      features['input_ids'] = tf.train.Feature(
          int64_list=tf.train.Int64List(value=list(input_ids)))
      features['label_ids'] = tf.train.Feature(
          int64_list=tf.train.Int64List(value=list([label_id])))
      tf_example = tf.train.Example(
          features=tf.train.Features(feature=features))
      writer.write(tf_example.SerializeToString())
    writer.close()

  def get_config(self):
    return {'seq_len': self.seq_len}


class MockQAModelSpec(object):

  def __init__(self, vocab_dir):
    self.seq_len = 384
    self.predict_batch_size = 8
    self.query_len = 64
    self.doc_stride = 128

    vocab_file = os.path.join(vocab_dir, 'vocab.txt')
    vocab = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', 'good', 'bad']
    with open(vocab_file, 'w') as f:
      f.write('\n'.join(vocab))
    self.tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case=True)

  def get_name_to_features(self, is_training):
    """Gets the dictionary describing the features."""
    name_to_features = {
        'input_ids': tf.io.FixedLenFeature([self.seq_len], tf.int64),
        'input_mask': tf.io.FixedLenFeature([self.seq_len], tf.int64),
        'segment_ids': tf.io.FixedLenFeature([self.seq_len], tf.int64),
    }

    if is_training:
      name_to_features['start_positions'] = tf.io.FixedLenFeature([], tf.int64)
      name_to_features['end_positions'] = tf.io.FixedLenFeature([], tf.int64)
    else:
      name_to_features['unique_ids'] = tf.io.FixedLenFeature([], tf.int64)

    return name_to_features

  def select_data_from_record(self, record):
    """Dispatches records to features and labels."""
    x, y = {}, {}
    for name, tensor in record.items():
      if name in ('start_positions', 'end_positions'):
        y[name] = tensor
      elif name == 'input_ids':
        x['input_word_ids'] = tensor
      elif name == 'segment_ids':
        x['input_type_ids'] = tensor
      else:
        x[name] = tensor
    return (x, y)

  def get_config(self):
    """Gets the configuration."""
    # Only preprocessing related variables are included.
    return {
        'seq_len': self.seq_len,
        'query_len': self.query_len,
        'doc_stride': self.doc_stride
    }

  def convert_examples_to_features(self, examples, is_training, output_fn,
                                   batch_size):
    """Converts examples to features and write them into TFRecord file."""
    return squad_lib.convert_examples_to_features(
        examples=examples,
        tokenizer=self.tokenizer,
        max_seq_length=self.seq_len,
        doc_stride=self.doc_stride,
        max_query_length=self.query_len,
        is_training=is_training,
        output_fn=output_fn,
        batch_size=batch_size)


class LoaderFunctionTest(tf.test.TestCase):

  def setUp(self):
    super(LoaderFunctionTest, self).setUp()
    self.model_spec = MockClassifierModelSpec()

  def test_load(self):
    tfrecord_file = self._get_tfrecord_file()
    meta_data_file = self._get_meta_data_file()
    dataset, meta_data = text_dataloader._load(tfrecord_file, meta_data_file,
                                               self.model_spec)
    for i, (input_ids, label_ids) in enumerate(dataset):
      self.assertEqual(i, 0)
      self.assertTrue((input_ids.numpy() == [0, 1, 2, 3]).all())
      self.assertTrue((label_ids.numpy() == [0]).all())
    self.assertEqual(meta_data['size'], 1)
    self.assertEqual(meta_data['num_classes'], 1)
    self.assertEqual(meta_data['index_to_label'], ['0'])

  def test_get_cache_filenames(self):
    tfrecord_file, meta_data_file, prefix = text_dataloader._get_cache_filenames(
        cache_dir='/tmp',
        model_spec=self.model_spec,
        data_name='train',
        is_training=True)
    self.assertTrue(tfrecord_file.startswith(prefix))
    self.assertTrue(meta_data_file.startswith(prefix))

    _, _, new_dir_prefix = text_dataloader._get_cache_filenames(
        cache_dir='/tmp1',
        model_spec=self.model_spec,
        data_name='train',
        is_training=True)
    self.assertNotEqual(new_dir_prefix, prefix)

    _, _, new_model_spec_prefix = text_dataloader._get_cache_filenames(
        cache_dir='/tmp',
        model_spec=MockClassifierModelSpec(seq_len=8),
        data_name='train',
        is_training=True)
    self.assertNotEqual(new_model_spec_prefix, prefix)

    _, _, new_data_name_prefix = text_dataloader._get_cache_filenames(
        cache_dir='/tmp',
        model_spec=self.model_spec,
        data_name='test',
        is_training=True)
    self.assertNotEqual(new_data_name_prefix, prefix)

    _, _, new_is_training_false_prefix = text_dataloader._get_cache_filenames(
        cache_dir='/tmp',
        model_spec=self.model_spec,
        data_name='train',
        is_training=False)
    self.assertNotEqual(new_is_training_false_prefix, prefix)

  def _get_tfrecord_file(self):
    tfrecord_file = os.path.join(self.get_temp_dir(), 'tmp.tfrecord')
    writer = tf.io.TFRecordWriter(tfrecord_file)
    input_ids = tf.train.Int64List(value=[0, 1, 2, 3])
    label_ids = tf.train.Int64List(value=[0])
    features = collections.OrderedDict()
    features['input_ids'] = tf.train.Feature(int64_list=input_ids)
    features['label_ids'] = tf.train.Feature(int64_list=label_ids)
    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())
    writer.close()
    return tfrecord_file

  def _get_meta_data_file(self):
    meta_data_file = os.path.join(self.get_temp_dir(), 'tmp_meta_data')
    meta_data = {'size': 1, 'num_classes': 1, 'index_to_label': ['0']}
    with tf.io.gfile.GFile(meta_data_file, 'w') as f:
      json.dump(meta_data, f)
    return meta_data_file


class TextClassifierDataLoaderTest(tf.test.TestCase):
  TEST_LABELS_AND_TEXT = (('pos', 'super good'), ('neg', 'really bad'))

  def _get_folder_path(self):
    folder_path = os.path.join(self.get_temp_dir(), 'random_text_dir')
    if os.path.exists(folder_path):
      return
    os.mkdir(folder_path)

    for label, text in self.TEST_LABELS_AND_TEXT:
      class_subdir = os.path.join(folder_path, label)
      os.mkdir(class_subdir)
      with open(os.path.join(class_subdir, '0.txt'), 'w') as f:
        f.write(text)
    return folder_path

  def _get_csv_file(self):
    csv_file = os.path.join(self.get_temp_dir(), 'tmp.csv')
    if os.path.exists(csv_file):
      return csv_file
    fieldnames = ['text', 'label']
    with open(csv_file, 'w') as f:
      writer = csv.DictWriter(f, fieldnames=fieldnames)
      writer.writeheader()
      for label, text in self.TEST_LABELS_AND_TEXT:
        writer.writerow({'text': text, 'label': label})
    return csv_file

  def test_split(self):
    ds = tf.data.Dataset.from_tensor_slices([[0, 1], [1, 1], [0, 0], [1, 0]])
    data = text_dataloader.TextClassifierDataLoader(ds, 4, 2, ['pos', 'neg'])
    train_data, test_data = data.split(0.5)

    self.assertEqual(train_data.size, 2)
    for i, elem in enumerate(train_data.dataset):
      self.assertTrue((elem.numpy() == np.array([i, 1])).all())
    self.assertEqual(train_data.num_classes, 2)
    self.assertEqual(train_data.index_to_label, ['pos', 'neg'])

    self.assertEqual(test_data.size, 2)
    for i, elem in enumerate(test_data.dataset):
      self.assertTrue((elem.numpy() == np.array([i, 0])).all())
    self.assertEqual(test_data.num_classes, 2)
    self.assertEqual(test_data.index_to_label, ['pos', 'neg'])

  def test_from_csv(self):
    csv_file = self._get_csv_file()
    model_spec = MockClassifierModelSpec()
    data = text_dataloader.TextClassifierDataLoader.from_csv(
        csv_file,
        text_column='text',
        label_column='label',
        model_spec=model_spec)
    self._test_data(data, model_spec)

  def test_from_folder(self):
    folder_path = self._get_folder_path()
    model_spec = MockClassifierModelSpec()
    data = text_dataloader.TextClassifierDataLoader.from_folder(
        folder_path, model_spec=model_spec)
    self._test_data(data, model_spec)

  def _test_data(self, data, model_spec):
    self.assertEqual(data.size, 2)
    self.assertEqual(data.num_classes, 2)
    self.assertEqual(data.index_to_label, ['neg', 'pos'])
    for input_ids, label in data.dataset:
      self.assertTrue(label.numpy() == 1 or label.numpy() == 0)
      actual_input_ids = [label.numpy()] * model_spec.seq_len
      self.assertTrue((input_ids.numpy() == actual_input_ids).all())


class QuestionAnswerDataLoaderTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(
      ('train-v1.1.json', True, False, 1),
      ('dev-v1.1.json', False, False, 8),
      ('train-v2.0.json', True, True, 2),
      ('dev-v2.0.json', False, True, 8),
  )
  def test_from_squad(self, test_file, is_training, version_2_with_negative,
                      size):

    path = test_util.get_test_data_path('squad_testdata')
    squad_path = os.path.join(path, test_file)
    model_spec = MockQAModelSpec(self.get_temp_dir())
    data = text_dataloader.QuestionAnswerDataLoader.from_squad(
        squad_path,
        model_spec,
        is_training=is_training,
        version_2_with_negative=version_2_with_negative)

    self.assertIsInstance(data, text_dataloader.QuestionAnswerDataLoader)
    self.assertEqual(data.size, size)
    self.assertEqual(data.version_2_with_negative, version_2_with_negative)
    self.assertEqual(data.squad_file, squad_path)

    if is_training:
      self.assertEmpty(data.features)
      self.assertEmpty(data.examples)
    else:
      self.assertNotEmpty(data.features)
      self.assertIsInstance(data.features[0], squad_lib.InputFeatures)

      self.assertEqual(len(data.features), len(data.examples))
      self.assertIsInstance(data.examples[0], squad_lib.SquadExample)


if __name__ == '__main__':
  tf.test.main()
