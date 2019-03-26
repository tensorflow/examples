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
import tensorflow as tf # TF2
from tensorflow_examples.models.densenet import densenet
from tensorflow_examples.models.densenet import utils
assert tf.__version__.startswith('2')


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
    self.autotune = tf.data.experimental.AUTOTUNE
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

  def keras_fit(self, train_dataset, test_dataset):
    self.model.compile(
        optimizer=self.optimizer, loss=self.loss_object, metrics=['accuracy'])
    history = self.model.fit(
        train_dataset, epochs=self.epochs, validation_data=test_dataset,
        verbose=2, callbacks=[tf.keras.callbacks.LearningRateScheduler(
            self.decay)])
    return (history.history['loss'][-1],
            history.history['accuracy'][-1],
            history.history['val_loss'][-1],
            history.history['val_accuracy'][-1])

  def train_step(self, image, label):
    """One train step.

    Args:
      image: Batch of images.
      label: corresponding label for the batch of images.
    """

    with tf.GradientTape() as tape:
      predictions = self.model(image, training=True)
      loss = self.loss_object(label, predictions)
      loss += sum(self.model.losses)
    gradients = tape.gradient(loss, self.model.trainable_variables)
    self.optimizer.apply_gradients(
        zip(gradients, self.model.trainable_variables))

    self.train_loss_metric(loss)
    self.train_acc_metric(label, predictions)

  def test_step(self, image, label):
    """One test step.

    Args:
      image: Batch of images.
      label: corresponding label for the batch of images.
    """

    predictions = self.model(image, training=False)
    loss = self.loss_object(label, predictions)

    self.test_loss_metric(loss)
    self.test_acc_metric(label, predictions)

  def custom_loop(self, train_dataset, test_dataset):
    """Custom training and testing loop.

    Args:
      train_dataset: Training dataset
      test_dataset: Testing dataset

    Returns:
      train_loss, train_accuracy, test_loss, test_accuracy
    """
    if self.enable_function:
      self.train_step = tf.function(self.train_step)
      self.test_step = tf.function(self.test_step)

    for epoch in range(self.epochs):
      self.optimizer.learning_rate = self.decay(epoch)

      for image, label in train_dataset:
        self.train_step(image, label)

      for test_image, test_label in test_dataset:
        self.test_step(test_image, test_label)

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
  kwargs = utils.flags_dict()
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
         data_dir=None):

  model = densenet.DenseNet(mode, growth_rate, output_classes, depth_of_model,
                            num_of_blocks, num_layers_in_each_block,
                            data_format, bottleneck, compression, weight_decay,
                            dropout_rate, pool_initial, include_top)
  train_obj = Train(epochs, enable_function, model)
  train_dataset, test_dataset, _ = utils.create_dataset(
      buffer_size, batch_size, data_format, data_dir)

  print('Training...')
  if train_mode == 'custom_loop':
    return train_obj.custom_loop(train_dataset, test_dataset)
  elif train_mode == 'keras_fit':
    return train_obj.keras_fit(train_dataset, test_dataset)


if __name__ == '__main__':
  utils.define_densenet_flags()
  app.run(run_main)
