# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""A demo of user defined task - use XOR problem as an example."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import os
import tempfile

from absl import app
from absl import flags
import numpy as np
import tensorflow as tf
from tensorflow_examples.lite.model_maker.core.data_util import data_util
from tensorflow_examples.lite.model_maker.core.data_util import dataloader
from tensorflow_examples.lite.model_maker.core.task import custom_model
from tensorflow_examples.lite.model_maker.core.task import model_util
from tflite_model_maker import config

from official.nlp import optimization

# pylint: disable=g-import-not-at-top,bare-except
try:
  from official.common import distribute_utils
except:
  from official.utils.misc import distribution_utils as distribute_utils
# pylint: enable=g-import-not-at-top,bare-except

FLAGS = flags.FLAGS


def define_flags():
  flags.DEFINE_string(
      'export_dir', None,
      'The directory to save exported TfLite/saved_model files.')
  flags.mark_flag_as_required('export_dir')


class DataLoader(dataloader.DataLoader):
  """DataLoader for XOR problem."""

  @classmethod
  def create(cls, spec, is_training=True, shuffle=False):
    inputs = tf.data.Dataset.from_tensor_slices([(0., 0), (0, 1), (1, 0),
                                                 (1, 1)])
    outputs = tf.data.Dataset.from_tensor_slices([0, 1, 1, 0])
    ds = tf.data.Dataset.zip((inputs, outputs))
    ds = ds.repeat(10)

    if shuffle:
      # Add some randomness in train/validation data split.
      ds = ds.shuffle(len(ds))

    ds = spec.preprocess_ds(ds, is_training=is_training)
    return cls(ds, len(ds))


class BinaryClassificationBaseSpec(abc.ABC):
  """Model spec for binary classification."""

  compat_tf_versions = (2,)

  def __init__(self, model_dir=None, strategy=None):
    self.model_dir = model_dir
    if not model_dir:
      self.model_dir = tempfile.mkdtemp()
    tf.compat.v1.logging.info('Checkpoints are stored in %s', self.model_dir)
    self.strategy = strategy or tf.distribute.get_strategy()

  @abc.abstractmethod
  def create_model(self):
    pass

  @abc.abstractmethod
  def run_classifier(self, model, epochs, train_ds, train_steps, validation_ds,
                     validation_steps, **kwargs):
    pass

  def preprocess_ds(self, ds, is_training=False):
    del is_training
    return ds


class Spec(BinaryClassificationBaseSpec):
  """Spec for XOR problem, contains a model with a single hidden layer."""

  def preprocess_ds(self, ds, is_training=False):

    @tf.function
    def _add_noise(x, y):
      """Add some random noise on the input sample."""
      return tf.random.normal(tf.shape(x), stddev=0.1) + x, y

    if is_training:
      ds = ds.map(_add_noise)
    return ds

  def create_model(self):
    model = tf.keras.Sequential()
    model.add(tf.keras.Input((2,)))
    model.add(tf.keras.layers.Dense(4, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    return model

  def run_classifier(self, model, epochs, train_ds, train_steps, validation_ds,
                     **kwargs):
    initial_lr = 0.1
    total_steps = train_steps * epochs
    warmup_steps = int(epochs * train_steps * 0.1)
    optimizer = optimization.create_optimizer(initial_lr, total_steps,
                                              warmup_steps)
    model.compile(
        optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    hist = model.fit(
        train_ds,
        steps_per_epoch=train_steps,
        validation_data=validation_ds,
        epochs=epochs,
        **kwargs)
    return hist


class BinaryClassifier(custom_model.CustomModel):
  """BinaryClassifier for train/inference and model export."""

  def __init__(self, spec, shuffle=True):
    assert isinstance(spec, BinaryClassificationBaseSpec)
    super(BinaryClassifier, self).__init__(spec, shuffle)

  def train(self, train_data, validation_data, epochs=10, batch_size=4):
    if len(train_data) < batch_size:
      raise ValueError('The size of the train_data (%d) couldn\'t be smaller '
                       'than batch_size (%d). To solve this problem, set '
                       'the batch_size smaller or increase the size of the '
                       'train_data.' % (len(train_data), batch_size))

    with distribute_utils.get_strategy_scope(self.model_spec.strategy):
      train_ds = train_data.gen_dataset(batch_size, is_training=True)
      train_steps = len(train_data) // batch_size
      validation_ds = validation_data.gen_dataset(
          batch_size, is_training=False) if validation_data else None

      self.model = self.model_spec.create_model()
      return self.model_spec.run_classifier(
          self.model,
          epochs,
          train_ds,
          train_steps,
          validation_ds,
          callbacks=self._keras_callbacks(self.model_spec.model_dir))

  def evaluate(self, data, batch_size=4):
    ds = data.gen_dataset(batch_size, is_training=False)
    return self.model.evaluate(ds, return_dict=True)

  def evaluate_tflite(self, tflite_filepath, data):
    ds = data.gen_dataset(batch_size=1, is_training=False)

    predictions, labels = [], []

    lite_runner = model_util.get_lite_runner(tflite_filepath, self.model_spec)
    for i, (feature, label) in enumerate(data_util.generate_elements(ds)):
      log_steps = 1000
      tf.compat.v1.logging.log_every_n(tf.compat.v1.logging.INFO,
                                       'Processing example: #%d\n%s', log_steps,
                                       i, feature)

      probability = lite_runner.run(feature)  # Shape: (batch=1, 1)
      probability = probability.flatten()[0]  # Get sclar value
      predictions.append(probability > 0.5)

      label = label[0]
      labels.append(label)

    predictions = np.array(predictions).astype(int)
    labels = np.array(labels).astype(int)

    return {'accuracy': (predictions == labels).mean()}


def train_xor_model(export_dir):
  """Use deep learning to solve XOR problem and return the test accuracy."""
  spec = Spec(model_dir=export_dir)

  data = DataLoader.create(spec, is_training=True)
  train_data, validation_data = data.split(0.8)

  classifier = BinaryClassifier(spec)
  classifier.train(train_data, validation_data)

  test_data = DataLoader.create(spec, is_training=False)
  eval_result = classifier.evaluate(test_data)
  eval_acc = eval_result['accuracy']
  print('Test accuracy: %f' % eval_acc)

  # Convert and quantize the model to tflite format.
  quantization = config.QuantizationConfig.for_int8(
      representative_data=test_data, quantization_steps=len(test_data))
  classifier.export(
      export_dir,
      quantization_config=quantization,
  )
  classifier.export(
      export_dir, export_format=(config.ExportFormat.SAVED_MODEL,))

  # Evaluate quantized tflite model.
  result = classifier.evaluate_tflite(
      os.path.join(export_dir, 'model.tflite'), test_data)
  test_acc = result['accuracy']
  print('Test accuracy on quantized TfLite model: %f' % test_acc)
  return test_acc


def main(_):
  train_xor_model(FLAGS.export_dir)


if __name__ == '__main__':
  define_flags()
  app.run(main)
