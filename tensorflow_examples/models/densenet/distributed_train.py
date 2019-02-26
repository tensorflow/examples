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
"""Densenet Training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import tensorflow as tf # TF2
import tensorflow_datasets as tfds
from tensorflow_examples.models.densenet import densenet
assert tf.__version__.startswith('2')

FLAGS = flags.FLAGS

flags.DEFINE_integer('buffer_size', 50000, 'Shuffle buffer size')
flags.DEFINE_integer('batch_size', 64, 'Batch Size')
flags.DEFINE_integer('epochs', 1, 'Number of epochs')
flags.DEFINE_boolean('enable_function', True, 'Enable Function?')
flags.DEFINE_string('data_dir', None, 'Directory to store the dataset')
flags.DEFINE_string('mode', 'from_depth', 'Deciding how to build the model')
flags.DEFINE_integer('depth_of_model', 3, 'Number of layers in the model')
flags.DEFINE_integer('growth_rate', 12, 'Filters to add per dense block')
flags.DEFINE_integer('num_of_blocks', 3, 'Number of dense blocks')
flags.DEFINE_integer('output_classes', 10, 'Number of classes in the dataset')
flags.DEFINE_integer('num_layers_in_each_block', -1,
                     'Number of layers in each dense block')
flags.DEFINE_string('data_format', 'channels_last',
                    'channels_last or channels_first')
flags.DEFINE_boolean('bottleneck', True, 'Add bottleneck blocks between layers')
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
flags.DEFINE_integer('num_gpu', 1, 'Number of GPUs to use')

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
    image = tf.image.resize_image_with_crop_or_pad(
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


class Train(object):
  """Train class.

  Args:
    epochs: Number of epochs
    enable_function: If True, wraps the train_step and test_step in tf.function
    model: Densenet model.
  """

  def __init__(self, epochs, enable_function, model):
    self.epochs = epochs
    self.enable_function = enable_function
    self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True)
    self.optimizer = tf.keras.optimizers.SGD(learning_rate=0.1,
                                             momentum=0.9, nesterov=True)
    self.train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
    self.train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(
        name='train_accuracy')
    self.test_loss_metric = tf.keras.metrics.Mean(name='test_loss')
    self.test_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(
        name='test_accuracy')
    self.model = model

  def decay(self, epoch):
    if epoch < 150:
      return 0.1
    if epoch >= 150 and epoch < 225:
      return 0.01
    if epoch >= 225:
      return 0.001

  def train_step(self, inputs):
    """One train step.

    Args:
      inputs: one batch input.
    """
    image, label = inputs
    with tf.GradientTape() as tape:
      predictions = self.model(image, training=True)
      loss = self.loss_object(label, predictions)
      loss += sum(self.model.losses)
    gradients = tape.gradient(loss, self.model.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients,
                                       self.model.trainable_variables))

    self.train_loss_metric(loss)
    self.train_acc_metric(label, predictions)

  def test_step(self, inputs):
    """One test step.

    Args:
      inputs: one batch input.
    """
    image, label = inputs
    predictions = self.model(image, training=False)
    loss = self.loss_object(label, predictions)
    loss += sum(self.model.losses)

    self.test_loss_metric(loss)
    self.test_acc_metric(label, predictions)

  def custom_loop(self, train_iterator, test_iterator,
                  num_train_steps_per_epoch, num_test_steps_per_epoch,
                  strategy):
    """Custom training and testing loop.

    Args:
      train_iterator: Training iterator created using strategy
      test_iterator: Testing iterator created using strategy
      num_train_steps_per_epoch: number of training steps in an epoch.
      num_test_steps_per_epoch: number of test steps in an epoch.
      strategy: Distribution strategy

    Returns:
      train_loss, train_accuracy, test_loss, test_accuracy
    """

    # this code is expected to change.
    distributed_train = lambda it: strategy.experimental_run(  # pylint: disable=g-long-lambda
        self.train_step, it)
    distributed_test = lambda it: strategy.experimental_run(  # pylint: disable=g-long-lambda
        self.test_step, it)

    if self.enable_function:
      distributed_train = tf.function(distributed_train)
      distributed_test = tf.function(distributed_test)

    for epoch in range(self.epochs):
      self.optimizer.learning_rate = self.decay(epoch)

      train_iterator.initialize()
      for _ in range(num_train_steps_per_epoch):
        distributed_train(train_iterator)

      test_iterator.initialize()
      for _ in range(num_test_steps_per_epoch):
        distributed_test(test_iterator)

      template = ('Epoch: {}, Train Loss: {}, Train Accuracy: {}, '
                  'Test Loss: {}, Test Accuracy: {}')

      print(
          template.format(epoch, self.train_loss_metric.result(),
                          self.train_acc_metric.result(),
                          self.test_loss_metric.result(),
                          self.test_acc_metric.result()))

      if epoch != self.epochs - 1:
        self.train_loss_metric.reset_states()
        self.train_acc_metric.reset_states()
        self.test_loss_metric.reset_states()
        self.test_acc_metric.reset_states()

    return (self.train_loss_metric.result().numpy(),
            self.train_acc_metric.result().numpy(),
            self.test_loss_metric.result().numpy(),
            self.test_acc_metric.result().numpy())


def run_main(argv):
  """Passes the flags to main.

  Args:
    argv: argv
  """
  del argv
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
      'train_mode': FLAGS.train_mode,
      'num_gpu': FLAGS.num_gpu
  }
  main(**kwargs)


def main(epochs,
         enable_function,
         buffer_size,
         batch_size,
         mode,
         growth_rate,
         output_classes,
         depth_of_model=None,
         num_of_blocks=None,
         num_layers_in_each_block=None,
         data_format='channels_last',
         bottleneck=True,
         compression=0.5,
         weight_decay=1e-4,
         dropout_rate=0.,
         pool_initial=False,
         include_top=True,
         train_mode='custom_loop',
         data_dir=None,
         num_gpu=1):

  devices = ['/device:GPU:{}'.format(i) for i in range(num_gpu)]
  strategy = tf.distribute.MirroredStrategy(devices)

  with strategy.scope():
    model = densenet.DenseNet(
        mode, growth_rate, output_classes, depth_of_model, num_of_blocks,
        num_layers_in_each_block, data_format, bottleneck, compression,
        weight_decay, dropout_rate, pool_initial, include_top)

    trainer = Train(epochs, enable_function, model)

    train_dataset, test_dataset, metadata = create_dataset(
        buffer_size, batch_size, data_format, data_dir)

    num_train_steps_per_epoch = metadata.splits[
        'train'].num_examples // batch_size
    num_test_steps_per_epoch = metadata.splits[
        'test'].num_examples // batch_size

    train_iterator = strategy.make_dataset_iterator(train_dataset)
    test_iterator = strategy.make_dataset_iterator(test_dataset)

    print('Training...')
    if train_mode == 'custom_loop':
      return trainer.custom_loop(train_iterator,
                                 test_iterator,
                                 num_train_steps_per_epoch,
                                 num_test_steps_per_epoch,
                                 strategy)
    elif train_mode == 'keras_fit':
      raise ValueError(
          '`tf.distribute.Strategy` does not support subclassed models yet.')
    else:
      raise ValueError(
          'Please enter either "keras_fit" or "custom_loop" as the argument.')


if __name__ == '__main__':
  app.run(run_main)
