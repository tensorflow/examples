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
"""Text dataloader."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import hashlib
import os
import random
import tempfile

from absl import logging
import tensorflow as tf
from tensorflow_examples.lite.model_maker.core import file_util
from tensorflow_examples.lite.model_maker.core.api import mm_export
from tensorflow_examples.lite.model_maker.core.data_util import dataloader
from tensorflow_examples.lite.model_maker.core.task import model_spec as ms

from official.nlp.bert import input_pipeline
from official.nlp.data import classifier_data_lib
from official.nlp.data import squad_lib


def _load(tfrecord_file, meta_data_file, model_spec, is_training=None):
  """Loads data from tfrecord file and metada file."""

  if is_training is None:
    name_to_features = model_spec.get_name_to_features()
  else:
    name_to_features = model_spec.get_name_to_features(is_training=is_training)

  dataset = input_pipeline.single_file_dataset(tfrecord_file, name_to_features)
  dataset = dataset.map(
      model_spec.select_data_from_record, num_parallel_calls=tf.data.AUTOTUNE)

  meta_data = file_util.load_json_file(meta_data_file)

  logging.info(
      'Load preprocessed data and metadata from %s and %s '
      'with size: %d', tfrecord_file, meta_data_file, meta_data['size'])
  return dataset, meta_data


def _get_cache_filenames(cache_dir, model_spec, data_name, is_training):
  """Gets cache tfrecord filename, metada filename and prefix of filenames."""
  hasher = hashlib.md5()
  hasher.update(data_name.encode('utf-8'))
  hasher.update(str(model_spec.get_config()).encode('utf-8'))
  hasher.update(str(is_training).encode('utf-8'))
  cache_prefix = os.path.join(cache_dir, hasher.hexdigest())
  cache_tfrecord_file = cache_prefix + '.tfrecord'
  cache_meta_data_file = cache_prefix + '_meta_data'

  return cache_tfrecord_file, cache_meta_data_file, cache_prefix


def _get_cache_info(cache_dir, data_name, model_spec, is_training):
  """Gets cache related information: whether is cached, related filenames."""
  if cache_dir is None:
    cache_dir = tempfile.mkdtemp()
  tfrecord_file, meta_data_file, file_prefix = _get_cache_filenames(
      cache_dir, model_spec, data_name, is_training)
  is_cached = tf.io.gfile.exists(tfrecord_file) and tf.io.gfile.exists(
      meta_data_file)

  return is_cached, tfrecord_file, meta_data_file, file_prefix


@mm_export('text_classifier.DataLoader')
class TextClassifierDataLoader(dataloader.ClassificationDataLoader):
  """DataLoader for text classifier."""

  @classmethod
  def from_folder(cls,
                  filename,
                  model_spec='average_word_vec',
                  is_training=True,
                  class_labels=None,
                  shuffle=True,
                  cache_dir=None):
    """Loads text with labels and preproecess text according to `model_spec`.

    Assume the text data of the same label are in the same subdirectory. each
    file is one text.

    Args:
      filename: Name of the file.
      model_spec: Specification for the model.
      is_training: Whether the loaded data is for training or not.
      class_labels: Class labels that should be considered. Name of the
        subdirectory not in `class_labels` will be ignored. If None, all the
        subdirectories will be considered.
      shuffle: boolean, if shuffle, random shuffle data.
      cache_dir: The cache directory to save preprocessed data. If None,
        generates a temporary directory to cache preprocessed data.

    Returns:
      TextDataset containing text, labels and other related info.
    """
    model_spec = ms.get(model_spec)
    data_root = os.path.abspath(filename)
    folder_name = os.path.basename(data_root)

    is_cached, tfrecord_file, meta_data_file, vocab_file = cls._get_cache_info(
        cache_dir, folder_name, model_spec, is_training)
    # If cached, directly loads data from cache directory.
    if is_cached:
      return cls._load_data(tfrecord_file, meta_data_file, model_spec)

    # Gets paths of all text.
    if class_labels:
      all_text_paths = []
      for class_label in class_labels:
        all_text_paths.extend(
            list(
                tf.io.gfile.glob(os.path.join(data_root, class_label) + r'/*')))
    else:
      all_text_paths = list(tf.io.gfile.glob(data_root + r'/*/*'))

    all_text_size = len(all_text_paths)
    if all_text_size == 0:
      raise ValueError('Text size is zero')

    if shuffle:
      random.shuffle(all_text_paths)

    # Gets label and its index.
    if class_labels:
      label_names = sorted(class_labels)
    else:
      label_names = sorted(
          name for name in os.listdir(data_root)
          if os.path.isdir(os.path.join(data_root, name)))

    # Generates text examples from folder.
    examples = []
    for i, path in enumerate(all_text_paths):
      with tf.io.gfile.GFile(path, 'r') as f:
        text = f.read()
      guid = '%s-%d' % (folder_name, i)
      label = os.path.basename(os.path.dirname(path))
      examples.append(classifier_data_lib.InputExample(guid, text, None, label))

    # Saves preprocessed data and other assets into files.
    cls._save_data(examples, model_spec, label_names, tfrecord_file,
                   meta_data_file, vocab_file, is_training)

    # Loads data from cache directory.
    return cls._load_data(tfrecord_file, meta_data_file, model_spec)

  @classmethod
  def from_csv(cls,
               filename,
               text_column,
               label_column,
               fieldnames=None,
               model_spec='average_word_vec',
               is_training=True,
               delimiter=',',
               quotechar='"',
               shuffle=False,
               cache_dir=None):
    """Loads text with labels from the csv file and preproecess text according to `model_spec`.

    Args:
      filename: Name of the file.
      text_column: String, Column name for input text.
      label_column: String, Column name for labels.
      fieldnames: A sequence, used in csv.DictReader. If fieldnames is omitted,
        the values in the first row of file f will be used as the fieldnames.
      model_spec: Specification for the model.
      is_training: Whether the loaded data is for training or not.
      delimiter: Character used to separate fields.
      quotechar: Character used to quote fields containing special characters.
      shuffle: boolean, if shuffle, random shuffle data.
      cache_dir: The cache directory to save preprocessed data. If None,
        generates a temporary directory to cache preprocessed data.

    Returns:
      TextDataset containing text, labels and other related info.
    """
    model_spec = ms.get(model_spec)
    csv_name = os.path.basename(filename)

    is_cached, tfrecord_file, meta_data_file, vocab_file = cls._get_cache_info(
        cache_dir, csv_name, model_spec, is_training)
    # If cached, directly loads data from cache directory.
    if is_cached:
      return cls._load_data(tfrecord_file, meta_data_file, model_spec)

    lines = cls._read_csv(filename, fieldnames, delimiter, quotechar)
    if shuffle:
      random.shuffle(lines)

    # Gets labels.
    label_set = set()
    for line in lines:
      label_set.add(line[label_column])
    label_names = sorted(label_set)

    # Generates text examples from csv file.
    examples = []
    for i, line in enumerate(lines):
      text, label = line[text_column], line[label_column]
      guid = '%s-%d' % (csv_name, i)
      examples.append(classifier_data_lib.InputExample(guid, text, None, label))

    # Saves preprocessed data and other assets into files.
    cls._save_data(examples, model_spec, label_names, tfrecord_file,
                   meta_data_file, vocab_file, is_training)

    # Loads data from cache directory.
    return cls._load_data(tfrecord_file, meta_data_file, model_spec)

  @classmethod
  def _load_data(cls, tfrecord_file, meta_data_file, model_spec):
    """Gets `TextClassifierDataLoader` object from tfrecord file and metadata file."""

    dataset, meta_data = _load(tfrecord_file, meta_data_file, model_spec)
    return TextClassifierDataLoader(dataset, meta_data['size'],
                                    meta_data['index_to_label'])

  @classmethod
  def _save_data(cls, examples, model_spec, label_names, tfrecord_file,
                 meta_data_file, vocab_file, is_training):
    """Saves preprocessed data and other assets into files."""
    # If needed, generates and saves vocabulary in vocab_file=None,
    if model_spec.need_gen_vocab and is_training:
      model_spec.gen_vocab(examples)
      model_spec.save_vocab(vocab_file)

    # Converts examples into preprocessed features and saves in tfrecord_file.
    model_spec.convert_examples_to_features(examples, tfrecord_file,
                                            label_names)

    # Generates and saves meta data in meta_data_file.
    meta_data = {
        'size': len(examples),
        'num_classes': len(label_names),
        'index_to_label': label_names
    }
    file_util.write_json_file(meta_data_file, meta_data)

  @classmethod
  def _get_cache_info(cls, cache_dir, data_name, model_spec, is_training):
    """Gets cache related information for text classifier."""
    is_cached, tfrecord_file, meta_data_file, file_prefix = _get_cache_info(
        cache_dir, data_name, model_spec, is_training)

    vocab_file = file_prefix + '_vocab'
    if is_cached:
      if model_spec.need_gen_vocab and is_training:
        model_spec.load_vocab(vocab_file)
      is_cached = True
    return is_cached, tfrecord_file, meta_data_file, vocab_file

  @classmethod
  def _read_csv(cls, input_file, fieldnames=None, delimiter=',', quotechar='"'):
    """Reads a separated value file."""
    with tf.io.gfile.GFile(input_file, 'r') as f:
      reader = csv.DictReader(
          f, fieldnames=fieldnames, delimiter=delimiter, quotechar=quotechar)
      lines = []
      for line in reader:
        lines.append(line)
      return lines


@mm_export('question_answer.DataLoader')
class QuestionAnswerDataLoader(dataloader.DataLoader):
  """DataLoader for question answering."""

  def __init__(self, dataset, size, version_2_with_negative, examples, features,
               squad_file):
    super(QuestionAnswerDataLoader, self).__init__(dataset, size)
    self.version_2_with_negative = version_2_with_negative
    self.examples = examples
    self.features = features
    self.squad_file = squad_file

  @classmethod
  def from_squad(cls,
                 filename,
                 model_spec,
                 is_training=True,
                 version_2_with_negative=False,
                 cache_dir=None):
    """Loads data in SQuAD format and preproecess text according to `model_spec`.

    Args:
      filename: Name of the file.
      model_spec: Specification for the model.
      is_training: Whether the loaded data is for training or not.
      version_2_with_negative: Whether it's SQuAD 2.0 format.
      cache_dir: The cache directory to save preprocessed data. If None,
        generates a temporary directory to cache preprocessed data.

    Returns:
      QuestionAnswerDataLoader object.
    """
    model_spec = ms.get(model_spec)
    file_base_name = os.path.basename(filename)
    is_cached, tfrecord_file, meta_data_file, _ = _get_cache_info(
        cache_dir, file_base_name, model_spec, is_training)
    # If cached, directly loads data from cache directory.
    if is_cached and is_training:
      dataset, meta_data = _load(tfrecord_file, meta_data_file, model_spec,
                                 is_training)
      return QuestionAnswerDataLoader(
          dataset=dataset,
          size=meta_data['size'],
          version_2_with_negative=meta_data['version_2_with_negative'],
          examples=[],
          features=[],
          squad_file=filename)

    meta_data, examples, features = cls._generate_tf_record_from_squad_file(
        filename, model_spec, tfrecord_file, is_training,
        version_2_with_negative)

    file_util.write_json_file(meta_data_file, meta_data)

    dataset, meta_data = _load(tfrecord_file, meta_data_file, model_spec,
                               is_training)
    return QuestionAnswerDataLoader(dataset, meta_data['size'],
                                    meta_data['version_2_with_negative'],
                                    examples, features, filename)

  @classmethod
  def _generate_tf_record_from_squad_file(cls,
                                          input_file_path,
                                          model_spec,
                                          output_path,
                                          is_training,
                                          version_2_with_negative=False):
    """Generates and saves training/validation data into a tf record file."""
    examples = squad_lib.read_squad_examples(
        input_file=input_file_path,
        is_training=is_training,
        version_2_with_negative=version_2_with_negative)
    writer = squad_lib.FeatureWriter(
        filename=output_path, is_training=is_training)

    features = []

    def _append_feature(feature, is_padding):
      if not is_padding:
        features.append(feature)
      writer.process_feature(feature)

    if is_training:
      batch_size = None
    else:
      batch_size = model_spec.predict_batch_size

    number_of_examples = model_spec.convert_examples_to_features(
        examples=examples,
        is_training=is_training,
        output_fn=writer.process_feature if is_training else _append_feature,
        batch_size=batch_size)
    writer.close()

    meta_data = {
        'size': number_of_examples,
        'version_2_with_negative': version_2_with_negative
    }

    if is_training:
      examples = []
    return meta_data, examples, features
