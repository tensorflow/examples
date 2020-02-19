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
import os
import random
import tempfile

import tensorflow as tf # TF2
from tensorflow_examples.lite.model_customization.core.data_util import dataloader
import tensorflow_examples.lite.model_customization.core.task.model_spec as ms
from official.nlp.bert import classifier_data_lib


def load(tfrecord_file, meta_data_file, model_spec):
  """Gets `TextClassifierDataLoader` object from tfrecord file and metadata file."""

  dataset, meta_data = dataloader.load(tfrecord_file, meta_data_file,
                                       model_spec)
  tf.compat.v1.logging.info(
      'Load preprocessed data and metadata from %s and %s '
      'with size: %d, num_classes: %d', tfrecord_file, meta_data_file,
      meta_data['size'], meta_data['num_classes'])
  return TextClassifierDataLoader(dataset, meta_data['size'],
                                  meta_data['num_classes'],
                                  meta_data['index_to_label'])


def save(examples, model_spec, label_names, tfrecord_file, meta_data_file,
         vocab_file, is_training):
  """Saves preprocessed data and other assets into files."""
  # If needed, generates and saves vocabulary in vocab_file=None,
  if model_spec.need_gen_vocab and is_training:
    model_spec.gen_vocab(examples)
    model_spec.save_vocab(vocab_file)

  # Converts examples into preprocessed features and saves in tfrecord_file.
  model_spec.convert_examples_to_features(examples, tfrecord_file, label_names)

  # Generates and saves meta data in meta_data_file.
  meta_data = {
      'size': len(examples),
      'num_classes': len(label_names),
      'index_to_label': label_names
  }
  dataloader.write_meta_data(meta_data_file, meta_data)


def get_cache_info(cache_dir, data_name, model_spec, is_training):
  """Gets cache related information: whether is cached, related filenames."""
  if cache_dir is None:
    cache_dir = tempfile.mkdtemp()
  tfrecord_file, meta_data_file, file_prefix = dataloader.get_cache_filenames(
      cache_dir, model_spec, data_name)
  vocab_file = file_prefix + '_vocab'

  is_cached = False
  if os.path.exists(tfrecord_file) and os.path.exists(meta_data_file):
    if model_spec.need_gen_vocab and is_training:
      model_spec.load_vocab(vocab_file)
    is_cached = True
  return is_cached, tfrecord_file, meta_data_file, vocab_file


def read_csv(input_file, fieldnames=None, delimiter=',', quotechar='"'):
  """Reads a separated value file."""
  with tf.io.gfile.GFile(input_file, 'r') as f:
    reader = csv.DictReader(
        f, fieldnames=fieldnames, delimiter=delimiter, quotechar=quotechar)
    lines = []
    for line in reader:
      lines.append(line)
    return lines


class TextClassifierDataLoader(dataloader.DataLoader):
  """DataLoader for text classifier."""

  def __init__(self, dataset, size, num_classes, index_to_label):
    super(TextClassifierDataLoader, self).__init__(dataset, size)
    self.num_classes = num_classes
    self.index_to_label = index_to_label

  def split(self, fraction, shuffle=True):
    """Splits dataset into two sub-datasets with the given fraction.

    Primarily used for splitting the data set into training and testing sets.

    Args:
      fraction: float, demonstrates the fraction of the first returned
        subdataset in the original data.
      shuffle: boolean, indicates whether to randomly shufflerandomly shuffle
        the data before splitting.

    Returns:
      The splitted two sub dataset.
    """
    if shuffle:
      ds = self.dataset.shuffle(
          buffer_size=self.size, reshuffle_each_iteration=False)
    else:
      ds = self.dataset

    train_size = int(self.size * fraction)
    trainset = TextClassifierDataLoader(
        ds.take(train_size), train_size, self.num_classes, self.index_to_label)

    test_size = self.size - train_size
    testset = TextClassifierDataLoader(
        ds.skip(train_size), test_size, self.num_classes, self.index_to_label)

    return trainset, testset

  @classmethod
  def from_folder(cls,
                  filename,
                  model_spec=ms.AverageWordVecModelSpec(),
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
    data_root = os.path.abspath(filename)
    folder_name = os.path.basename(data_root)

    is_cached, tfrecord_file, meta_data_file, vocab_file = get_cache_info(
        cache_dir, folder_name, model_spec, is_training)
    # If cached, directly loads data from cache directory.
    if is_cached:
      return load(tfrecord_file, meta_data_file, model_spec)

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
    save(examples, model_spec, label_names, tfrecord_file, meta_data_file,
         vocab_file, is_training)

    # Loads data from cache directory.
    return load(tfrecord_file, meta_data_file, model_spec)

  @classmethod
  def from_csv(cls,
               filename,
               text_column,
               label_column,
               fieldnames=None,
               model_spec=ms.AverageWordVecModelSpec(),
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
    csv_name = os.path.basename(filename)

    is_cached, tfrecord_file, meta_data_file, vocab_file = get_cache_info(
        cache_dir, csv_name, model_spec, is_training)
    # If cached, directly loads data from cache directory.
    if is_cached:
      return load(tfrecord_file, meta_data_file, model_spec)

    lines = read_csv(filename, fieldnames, delimiter, quotechar)
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
    save(examples, model_spec, label_names, tfrecord_file, meta_data_file,
         vocab_file, is_training)

    # Loads data from cache directory.
    return load(tfrecord_file, meta_data_file, model_spec)
