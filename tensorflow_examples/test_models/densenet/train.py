"""Densenet Training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import tensorflow as tf # TF2
import tensorflow_datasets as tfds
from tensorflow_examples.test_models.densenet import densenet
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
                    'Use either keras fit or custom loops')

AUTOTUNE = tf.data.experimental.AUTOTUNE


def scale(image, label):
  image = tf.cast(image, tf.float32)
  image /= 255

  return image, label


def create_dataset(buffer_size, batch_size, data_dir=None):
  """Creates a tf.data Dataset.

  Args:
    buffer_size: Shuffle buffer size.
    batch_size: Batch size
    data_dir: directory to store the dataset.

  Returns:
    train dataset, test dataset
  """
  dataset, _ = tfds.load(
      'cifar10', data_dir=data_dir, as_supervised=True, with_info=True)
  train_dataset, test_dataset = dataset['train'], dataset['test']

  train_dataset = train_dataset.map(scale, num_parallel_calls=AUTOTUNE)
  train_dataset = train_dataset.shuffle(buffer_size).batch(batch_size)

  test_dataset = test_dataset.map(
      scale, num_parallel_calls=AUTOTUNE).batch(batch_size)

  return train_dataset, test_dataset


class Train(object):
  """Train class.

  Args:
    epochs: Number of epochs
    enable_function: If True, wraps the train_step and test_step in tf.function
  """

  def __init__(self, epochs, enable_function):
    self.epochs = epochs
    self.enable_function = enable_function
    self.autotune = tf.data.experimental.AUTOTUNE
    self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True)
    self.optimizer = tf.keras.optimizers.Adam(1e-4)
    self.train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
    self.train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(
        name='train_accuracy')
    self.test_loss_metric = tf.keras.metrics.Mean(name='test_loss')
    self.test_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(
        name='test_accuracy')

  def keras_fit(self, train_dataset, test_dataset, model):
    model.compile(
        optimizer=self.optimizer, loss=self.loss_object, metrics=['accuracy'])
    history = model.fit(
        train_dataset, epochs=self.epochs, validation_data=test_dataset)
    return (history.history['loss'][-1], history.history['acc'][-1],
            history.history['val_loss'][-1], history.history['val_acc'][-1])

  def train_step(self, image, label, model):
    with tf.GradientTape() as tape:
      predictions = model(image, training=True)
      loss = self.loss_object(label, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    self.train_loss_metric(loss)
    self.train_acc_metric(label, predictions)

  def test_step(self, image, label, model):
    predictions = model(image, training=False)
    loss = self.loss_object(label, predictions)

    self.test_loss_metric(loss)
    self.test_acc_metric(label, predictions)

  def custom_loop(self, train_dataset, test_dataset, model):
    """Custom training and testing loop.

    Args:
      train_dataset: Training dataset
      test_dataset: Testing dataset
      model: Model to train and test

    Returns:
      train_loss, train_accuracy, test_loss, test_accuracy
    """
    if self.enable_function:
      self.train_step = tf.function(self.train_step)
      self.test_step = tf.function(self.test_step)

    for epoch in range(self.epochs):
      for image, label in train_dataset:
        self.train_step(image, label, model)

      for test_image, test_label in test_dataset:
        self.test_step(test_image, test_label, model)

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

    return (self.train_loss_metric.result(), self.train_acc_metric.result(),
            self.test_loss_metric.result(), self.test_acc_metric.result())


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
      'train_mode': FLAGS.train_mode
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
         data_dir=None):

  train_obj = Train(epochs, enable_function)
  train_dataset, test_dataset = create_dataset(buffer_size, batch_size,
                                               data_dir)
  model = densenet.DenseNet(mode, growth_rate, output_classes, depth_of_model,
                            num_of_blocks, num_layers_in_each_block,
                            data_format, bottleneck, compression, weight_decay,
                            dropout_rate, pool_initial, include_top)
  print('Training...')
  if train_mode == 'custom_loop':
    return train_obj.custom_loop(train_dataset, test_dataset, model)
  elif train_mode == 'keras_fit':
    return train_obj.keras_fit(train_dataset, test_dataset, model)


if __name__ == '__main__':
  app.run(run_main)
