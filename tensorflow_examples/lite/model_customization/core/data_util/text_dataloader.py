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

import os
import random
import tensorflow as tf # TF2
from tensorflow_examples.lite.model_customization.core.data_util import dataloader


def load_text(path):
  raw_text = tf.io.read_file(path)
  return raw_text


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
  def from_folder(cls, filename, class_labels=None, shuffle=True):
    """Text analysis for text classification load text with labels.

    Assume the text data of the same label are in the same subdirectory. each
    file is one text.

    Args:
      filename: Name of the file.
      class_labels: Class labels that should be considered. Name of the
        subdirectory not in `class_labels` will be ignored. If None, all the
        subdirectories will be considered.
      shuffle: boolean, if shuffle, random shuffle data.

    Returns:
      TextDataset containing images, labels and other related info.
    """
    data_root = os.path.abspath(filename)

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
    all_label_size = len(label_names)
    label_to_index = dict(
        (name, index) for index, name in enumerate(label_names))
    all_text_labels = [
        label_to_index[os.path.basename(os.path.dirname(path))]
        for path in all_text_paths
    ]

    # Gets raw data.
    path_ds = tf.data.Dataset.from_tensor_slices(all_text_paths)

    autotune = tf.data.experimental.AUTOTUNE
    text_ds = path_ds.map(load_text, num_parallel_calls=autotune)

    # Loads label.
    label_ds = tf.data.Dataset.from_tensor_slices(
        tf.cast(all_text_labels, tf.int64))

    # Creates a dataset if (text, label) pairs.
    text_label_ds = tf.data.Dataset.zip((text_ds, label_ds))

    tf.compat.v1.logging.info(
        'load text from %s with size: %d, num_label: %d, labels: %s', data_root,
        all_text_size, all_label_size, ', '.join(label_names))
    return TextClassifierDataLoader(text_label_ds, all_text_size,
                                    all_label_size, label_names)
