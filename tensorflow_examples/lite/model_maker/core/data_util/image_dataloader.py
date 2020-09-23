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
"""Image dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_examples.lite.model_maker.core.data_util import dataloader


def load_image(path):
  """Loads image."""
  image_raw = tf.io.read_file(path)
  image_tensor = tf.cond(
      tf.image.is_jpeg(image_raw),
      lambda: tf.image.decode_jpeg(image_raw, channels=3),
      lambda: tf.image.decode_png(image_raw, channels=3))
  return image_tensor


def create_data(name, data, info, num_classes, label_names):
  """Creates an ImageClassifierDataLoader object from tfds data."""
  if name not in data:
    return None
  data = data[name]
  data = data.map(lambda a: (a['image'], a['label']))
  size = info.splits[name].num_examples
  return ImageClassifierDataLoader(data, size, num_classes, label_names)


class ImageClassifierDataLoader(dataloader.DataLoader):
  """DataLoader for image classifier."""

  def __init__(self, dataset, size, num_classes, index_to_label):
    super(ImageClassifierDataLoader, self).__init__(dataset, size)
    self.num_classes = num_classes
    self.index_to_label = index_to_label

  def split(self, fraction):
    """Splits dataset into two sub-datasets with the given fraction.

    Primarily used for splitting the data set into training and testing sets.

    Args:
      fraction: float, demonstrates the fraction of the first returned
        subdataset in the original data.

    Returns:
      The splitted two sub dataset.
    """
    ds = self.dataset

    train_size = int(self.size * fraction)
    trainset = ImageClassifierDataLoader(
        ds.take(train_size), train_size, self.num_classes, self.index_to_label)

    test_size = self.size - train_size
    testset = ImageClassifierDataLoader(
        ds.skip(train_size), test_size, self.num_classes, self.index_to_label)

    return trainset, testset

  @classmethod
  def from_folder(cls, filename, shuffle=True):
    """Image analysis for image classification load images with labels.

    Assume the image data of the same label are in the same subdirectory.

    Args:
      filename: Name of the file.
      shuffle: boolean, if shuffle, random shuffle data.

    Returns:
      ImageDataset containing images and labels and other related info.
    """
    data_root = os.path.abspath(filename)

    # Assumes the image data of the same label are in the same subdirectory,
    # gets image path and label names.
    all_image_paths = list(tf.io.gfile.glob(data_root + r'/*/*'))
    all_image_size = len(all_image_paths)
    if all_image_size == 0:
      raise ValueError('Image size is zero')

    if shuffle:
      # Random shuffle data.
      random.shuffle(all_image_paths)

    label_names = sorted(
        name for name in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, name)))
    all_label_size = len(label_names)
    label_to_index = dict(
        (name, index) for index, name in enumerate(label_names))
    all_image_labels = [
        label_to_index[os.path.basename(os.path.dirname(path))]
        for path in all_image_paths
    ]

    path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)

    autotune = tf.data.experimental.AUTOTUNE
    image_ds = path_ds.map(load_image, num_parallel_calls=autotune)

    # Loads label.
    label_ds = tf.data.Dataset.from_tensor_slices(
        tf.cast(all_image_labels, tf.int64))

    # Creates  a dataset if (image, label) pairs.
    image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))

    tf.compat.v1.logging.info(
        'Load image with size: %d, num_label: %d, labels: %s.', all_image_size,
        all_label_size, ', '.join(label_names))
    return ImageClassifierDataLoader(image_label_ds, all_image_size,
                                     all_label_size, label_names)

  @classmethod
  def from_tfds(cls, name):
    """Loads data from tensorflow_datasets."""
    data, info = tfds.load(name, with_info=True)
    if 'label' not in info.features:
      raise ValueError('info.features need to contain \'label\' key.')
    num_classes = info.features['label'].num_classes
    label_names = info.features['label'].names

    train_data = create_data('train', data, info, num_classes, label_names)
    validation_data = create_data('validation', data, info, num_classes,
                                  label_names)
    test_data = create_data('test', data, info, num_classes, label_names)
    return train_data, validation_data, test_data
