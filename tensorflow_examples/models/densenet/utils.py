# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Densenet utils.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
import tensorflow as tf # TF2
import tensorflow_datasets as tfds
assert tf.__version__.startswith('2')

FLAGS = flags.FLAGS


def define_densenet_flags():
  """Defining all the necessary flags."""
  flags.DEFINE_integer('buffer_size', 50000, 'Shuffle buffer size')
  flags.DEFINE_integer('batch_size', 64, 'Batch Size')
  flags.DEFINE_integer('epochs', 1, 'Number of epochs')
  flags.DEFINE_boolean('enable_function', True, 'Enable Function?')
  flags.DEFINE_string('data_dir', None, 'Directory to store the dataset')
  flags.DEFINE_string('mode', 'from_depth', 'Deciding how to build the model')
  flags.DEFINE_integer('depth_of_model', 7, 'Number of layers in the model')
  flags.DEFINE_integer('growth_rate', 12, 'Filters to add per dense block')
  flags.DEFINE_integer('num_of_blocks', 3, 'Number of dense blocks')
  flags.DEFINE_integer('output_classes', 10, 'Number of classes in the dataset')
  flags.DEFINE_integer('num_layers_in_each_block', -1,
                       'Number of layers in each dense block')
  flags.DEFINE_string('data_format', 'channels_last',
                      'channels_last or channels_first')
  flags.DEFINE_boolean('bottleneck', True,
                       'Add bottleneck blocks between layers')
  flags.DEFINE_float(
      'compression', 0.5,
      'reducing the number of inputs(filters) to the transition block.')
  flags.DEFINE_float('weight_decay', 1e-4, 'weight decay')
  flags.DEFINE_float('dropout_rate', 0., 'dropout rate')
  flags.DEFINE_boolean(
      'pool_initial', False,
      'If True add a conv => maxpool block at the start. Used for Imagenet')
  flags.DEFINE_boolean('include_top', True, 'Include the classifier layer')
  flags.DEFINE_string('train_mode', 'custom_loop',
                      'Use either "keras_fit" or "custom_loop"')

AUTOTUNE = tf.data.experimental.AUTOTUNE

CIFAR_MEAN = [125.3, 123.0, 113.9]
CIFAR_STD = [63.0, 62.1, 66.7]

HEIGHT = 32
WIDTH = 32


class Preprocess(object):
  """Preprocess images.

  Args:
    data_format: channels_first or channels_last
  """

  def __init__(self, data_format, train):
    self.data_format = data_format
    self.train = train

  def __call__(self, image, label):
    image = tf.cast(image, tf.float32)

    if self.train:
      image = tf.image.random_flip_left_right(image)
      image = self.random_jitter(image)

    image = (image - CIFAR_MEAN) / CIFAR_STD

    if self.data_format == 'channels_first':
      image = tf.transpose(image, [2, 0, 1])

    return image, label

  def random_jitter(self, image):
    # add 4 pixels on each side; image_size == (36 x 36)
    image = tf.image.resize_with_crop_or_pad(
        image, HEIGHT + 8, WIDTH + 8)

    image = tf.image.random_crop(image, size=[HEIGHT, WIDTH, 3])

    return image


def create_dataset(buffer_size, batch_size, data_format, data_dir=None):
  """Creates a tf.data Dataset.

  Args:
    buffer_size: Shuffle buffer size.
    batch_size: Batch size
    data_format: channels_first or channels_last
    data_dir: directory to store the dataset.

  Returns:
    train dataset, test dataset, metadata
  """

  preprocess_train = Preprocess(data_format, train=True)
  preprocess_test = Preprocess(data_format, train=False)

  dataset, metadata = tfds.load(
      'cifar10', data_dir=data_dir, as_supervised=True, with_info=True)
  train_dataset, test_dataset = dataset['train'], dataset['test']

  train_dataset = train_dataset.map(
      preprocess_train, num_parallel_calls=AUTOTUNE)
  train_dataset = train_dataset.shuffle(buffer_size).batch(batch_size)
  train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)

  test_dataset = test_dataset.map(
      preprocess_test, num_parallel_calls=AUTOTUNE).batch(batch_size)
  test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

  return train_dataset, test_dataset, metadata


def flags_dict():
  """Define the flags.

  Returns:
    Command line arguments as Flags.
  """

  kwargs = {
      'epochs': FLAGS.epochs,
      'enable_function': FLAGS.enable_function,
      'buffer_size': FLAGS.buffer_size,
      'batch_size': FLAGS.batch_size,
      'mode': FLAGS.mode,
      'depth_of_model': FLAGS.depth_of_model,
      'growth_rate': FLAGS.growth_rate,
      'num_of_blocks': FLAGS.num_of_blocks,
      'output_classes': FLAGS.output_classes,
      'num_layers_in_each_block': FLAGS.num_layers_in_each_block,
      'data_format': FLAGS.data_format,
      'bottleneck': FLAGS.bottleneck,
      'compression': FLAGS.compression,
      'weight_decay': FLAGS.weight_decay,
      'dropout_rate': FLAGS.dropout_rate,
      'pool_initial': FLAGS.pool_initial,
      'include_top': FLAGS.include_top,
      'train_mode': FLAGS.train_mode
  }
  return kwargs


def get_cifar10_kwargs():
  return {'epochs': 1, 'enable_function': True, 'buffer_size': 50000,
          'batch_size': 64, 'depth_of_model': 40, 'growth_rate': 12,
          'num_of_blocks': 3, 'output_classes': 10, 'mode': 'from_depth',
          'data_format': 'channels_last', 'dropout_rate': 0.}
