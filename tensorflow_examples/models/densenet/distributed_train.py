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
from tensorflow_examples.models.densenet import densenet
from tensorflow_examples.models.densenet import utils
assert tf.__version__.startswith('2')

FLAGS = flags.FLAGS

# if additional flags are needed, define it here.
flags.DEFINE_integer('num_gpu', 1, 'Number of GPUs to use')


class Train(object):
  """Train class.

  Args:
    epochs: Number of epochs
    enable_function: If True, wraps the train_step and test_step in tf.function
    model: Densenet model.
    batch_size: Batch size.
    strategy: Distribution strategy in use.
  """

  def __init__(self, epochs, enable_function, model, batch_size, strategy):
    self.epochs = epochs
    self.batch_size = batch_size
    self.enable_function = enable_function
    self.strategy = strategy
    self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
    self.optimizer = tf.keras.optimizers.SGD(learning_rate=0.1,
                                             momentum=0.9, nesterov=True)
    self.train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(
        name='train_accuracy')
    self.test_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(
        name='test_accuracy')
    self.test_loss_metric = tf.keras.metrics.Sum(name='test_loss')
    self.model = model

  def decay(self, epoch):
    if epoch < 150:
      return 0.1
    if epoch >= 150 and epoch < 225:
      return 0.01
    if epoch >= 225:
      return 0.001

  def compute_loss(self, label, predictions):
    loss = tf.reduce_sum(self.loss_object(label, predictions)) * (
        1. / self.batch_size)
    loss += (sum(self.model.losses) * 1. / self.strategy.num_replicas_in_sync)
    return loss

  def train_step(self, inputs):
    """One train step.

    Args:
      inputs: one batch input.

    Returns:
      loss: Scaled loss.
    """

    image, label = inputs
    with tf.GradientTape() as tape:
      predictions = self.model(image, training=True)
      loss = self.compute_loss(label, predictions)
    gradients = tape.gradient(loss, self.model.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients,
                                       self.model.trainable_variables))

    self.train_acc_metric(label, predictions)
    return loss

  def test_step(self, inputs):
    """One test step.

    Args:
      inputs: one batch input.
    """
    image, label = inputs
    predictions = self.model(image, training=False)

    unscaled_test_loss = self.loss_object(label, predictions) + sum(
        self.model.losses)

    self.test_acc_metric(label, predictions)
    self.test_loss_metric(unscaled_test_loss)

  def custom_loop(self, train_dist_dataset, test_dist_dataset,
                  strategy):
    """Custom training and testing loop.

    Args:
      train_dist_dataset: Training dataset created using strategy.
      test_dist_dataset: Testing dataset created using strategy.
      strategy: Distribution strategy.

    Returns:
      train_loss, train_accuracy, test_loss, test_accuracy
    """

    def distributed_train_epoch(ds):
      total_loss = 0.0
      num_train_batches = 0.0
      for one_batch in ds:
        per_replica_loss = strategy.experimental_run_v2(
            self.train_step, args=(one_batch,))
        total_loss += strategy.reduce(
            tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None)
        num_train_batches += 1
      return total_loss, num_train_batches

    def distributed_test_epoch(ds):
      num_test_batches = 0.0
      for one_batch in ds:
        strategy.experimental_run_v2(
            self.test_step, args=(one_batch,))
        num_test_batches += 1
      return self.test_loss_metric.result(), num_test_batches

    if self.enable_function:
      distributed_train_epoch = tf.function(distributed_train_epoch)
      distributed_test_epoch = tf.function(distributed_test_epoch)

    for epoch in range(self.epochs):
      self.optimizer.learning_rate = self.decay(epoch)

      train_total_loss, num_train_batches = distributed_train_epoch(
          train_dist_dataset)
      test_total_loss, num_test_batches = distributed_test_epoch(
          test_dist_dataset)

      template = ('Epoch: {}, Train Loss: {}, Train Accuracy: {}, '
                  'Test Loss: {}, Test Accuracy: {}')

      print(
          template.format(epoch,
                          train_total_loss / num_train_batches,
                          self.train_acc_metric.result(),
                          test_total_loss / num_test_batches,
                          self.test_acc_metric.result()))

      if epoch != self.epochs - 1:
        self.train_acc_metric.reset_states()
        self.test_acc_metric.reset_states()

    return (train_total_loss / num_train_batches,
            self.train_acc_metric.result().numpy(),
            test_total_loss / num_test_batches,
            self.test_acc_metric.result().numpy())


def run_main(argv):
  """Passes the flags to main.

  Args:
    argv: argv
  """
  del argv
  kwargs = utils.flags_dict()
  kwargs.update({'num_gpu': FLAGS.num_gpu})
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

  train_dataset, test_dataset, _ = utils.create_dataset(
      buffer_size, batch_size, data_format, data_dir)

  with strategy.scope():
    model = densenet.DenseNet(
        mode, growth_rate, output_classes, depth_of_model, num_of_blocks,
        num_layers_in_each_block, data_format, bottleneck, compression,
        weight_decay, dropout_rate, pool_initial, include_top)

    trainer = Train(epochs, enable_function, model, batch_size, strategy)

    train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
    test_dist_dataset = strategy.experimental_distribute_dataset(test_dataset)

    print('Training...')
    if train_mode == 'custom_loop':
      return trainer.custom_loop(train_dist_dataset,
                                 test_dist_dataset,
                                 strategy)
    elif train_mode == 'keras_fit':
      raise ValueError(
          '`tf.distribute.Strategy` does not support subclassed models yet.')
    else:
      raise ValueError(
          'Please enter either "keras_fit" or "custom_loop" as the argument.')


if __name__ == '__main__':
  utils.define_densenet_flags()
  app.run(run_main)
