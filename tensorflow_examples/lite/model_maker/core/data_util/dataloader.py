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
"""Common Dataset used for tasks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
from typing import Optional

import tensorflow as tf


def shard(ds, input_pipeline_context):
  # The dataset is always sharded by number of hosts.
  # num_input_pipelines is the number of hosts rather than number of cores.
  if (input_pipeline_context and
      input_pipeline_context.num_input_pipelines > 1):
    ds = ds.shard(input_pipeline_context.num_input_pipelines,
                  input_pipeline_context.input_pipeline_id)
  return ds


class DataLoader(object):
  """This class provides generic utilities for loading customized domain data that will be used later in model retraining.

  For different ML problems or tasks, such as image classification, text
  classification etc., a subclass is provided to handle task-specific data
  loading requirements.
  """

  def __init__(self, dataset, size=None):
    """Init function for class `DataLoader`.

    In most cases, one should use helper functions like `from_folder` to create
    an instance of this class.

    Args:
      dataset: A tf.data.Dataset object that contains a potentially large set of
        elements, where each element is a pair of (input_data, target). The
        `input_data` means the raw input data, like an image, a text etc., while
        the `target` means some ground truth of the raw input data, such as the
        classification label of the image etc.
      size: The size of the dataset. tf.data.Dataset donesn't support a function
        to get the length directly since it's lazy-loaded and may be infinite.
    """
    self._dataset = dataset
    self._size = size

  @property
  def size(self) -> Optional[int]:
    """Returns the size of the dataset.

    Note that this function may return None becuase the exact size of the
    dataset isn't a necessary parameter to create an instance of this class,
    and tf.data.Dataset donesn't support a function to get the length directly
    since it's lazy-loaded and may be infinite.
    In most cases, however, when an instance of this class is created by helper
    functions like 'from_folder', the size of the dataset will be preprocessed,
    and this function can return an int representing the size of the dataset.
    """
    return self._size

  def gen_dataset(self,
                  batch_size=1,
                  is_training=False,
                  shuffle=False,
                  input_pipeline_context=None,
                  preprocess=None,
                  drop_remainder=False):
    """Generate a shared and batched tf.data.Dataset for training/evaluation.

    Args:
      batch_size: A integer, the returned dataset will be batched by this size.
      is_training: A boolean, when True, the returned dataset will be optionally
        shuffled and repeated as an endless dataset.
      shuffle: A boolean, when True, the returned dataset will be shuffled to
        create randomness during model training.
      input_pipeline_context: A InputContext instance, used to shared dataset
        among multiple workers when distribution strategy is used.
      preprocess: A function taking three arguments in order, feature, label and
        boolean is_training.
      drop_remainder: boolean, whether the finaly batch drops remainder.

    Returns:
      A TF dataset ready to be consumed by Keras model.
    """
    ds = self._dataset
    ds = shard(ds, input_pipeline_context)

    if preprocess:
      preprocess = functools.partial(preprocess, is_training=is_training)
      ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    if is_training:
      if shuffle:
        # Shuffle size should be bigger than the batch_size. Otherwise it's only
        # shuffling within the batch, which equals to not having shuffle.
        buffer_size = 3 * batch_size
        # But since we are doing shuffle before repeat, it doesn't make sense to
        # shuffle more than total available entries.
        # TODO(wangtz): Do we want to do shuffle before / after repeat?
        # Shuffle after repeat will give a more randomized dataset and mix the
        # epoch boundary: https://www.tensorflow.org/guide/data
        if self._size:
          buffer_size = min(self._size, buffer_size)
        ds = ds.shuffle(buffer_size=buffer_size)

    ds = ds.batch(batch_size, drop_remainder=drop_remainder)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    # TODO(b/171449557): Consider converting ds to distributed ds here.
    return ds

  def __len__(self):
    if self._size is not None:
      return self._size
    else:
      return len(self._dataset)

  def split(self, fraction):
    """Splits dataset into two sub-datasets with the given fraction.

    Primarily used for splitting the data set into training and testing sets.

    Args:
      fraction: float, demonstrates the fraction of the first returned
        subdataset in the original data.

    Returns:
      The splitted two sub datasets.
    """
    return self._split(fraction)

  def _split(self, fraction, *args):
    """Actual implementation for `split` method and returns sub-class instances.

    Child DataLoader, if requires additional constructor arguments, should
      implement their own `split` method by calling `_split` with all arguments
      to the constructor.

    Args:
      fraction: float, demonstrates the fraction of the first returned
        subdataset in the original data.
      *args: additional arguments passed to the sub-class constructor.

    Returns:
      The splitted two sub datasets.
    """
    assert (fraction > 0 and fraction < 1)

    ds = self._dataset

    train_size = int(self._size * fraction)
    trainset = self.__class__(ds.take(train_size), train_size, *args)

    test_size = self._size - train_size
    testset = self.__class__(ds.skip(train_size), test_size, *args)

    return trainset, testset


class ClassificationDataLoader(DataLoader):
  """DataLoader for classification models."""

  def __init__(self, dataset, size, index_to_label):
    super(ClassificationDataLoader, self).__init__(dataset, size)
    self.index_to_label = index_to_label

  @property
  def num_classes(self):
    return len(self.index_to_label)

  def split(self, fraction):
    """Splits dataset into two sub-datasets with the given fraction.

    Primarily used for splitting the data set into training and testing sets.

    Args:
      fraction: float, demonstrates the fraction of the first returned
        subdataset in the original data.

    Returns:
      The splitted two sub datasets.
    """
    return self._split(fraction, self.index_to_label)
