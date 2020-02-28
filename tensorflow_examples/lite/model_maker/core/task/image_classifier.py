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
"""ImageClassier class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf # TF2

from tensorflow_examples.lite.model_maker.core import compat
from tensorflow_examples.lite.model_maker.core import model_export_format as mef
from tensorflow_examples.lite.model_maker.core.task import classification_model
from tensorflow_examples.lite.model_maker.core.task import hub_loader
from tensorflow_examples.lite.model_maker.core.task import model_spec as ms

from tensorflow_hub.tools.make_image_classifier import make_image_classifier_lib as lib


def create(train_data,
           model_export_format=mef.ModelExportFormat.TFLITE,
           model_spec=ms.mobilenet_v2_spec,
           shuffle=False,
           validation_data=None,
           batch_size=None,
           epochs=None,
           train_whole_model=None,
           dropout_rate=None,
           learning_rate=None,
           momentum=None):
  """Loads data and retrains the model based on data for image classification.

  Args:
    train_data: Training data.
    model_export_format: Model export format such as saved_model / tflite.
    model_spec: Specification for the model.
    shuffle: Whether the data should be shuffled.
    validation_data: Validation data. If None, skips validation process.
    batch_size: Number of samples per training step.
    epochs: Number of epochs for training.
    train_whole_model: If true, the Hub module is trained together with the
      classification layer on top. Otherwise, only train the top classification
      layer.
    dropout_rate: the rate for dropout.
    learning_rate: a Python float forwarded to the optimizer.
    momentum: a Python float forwarded to the optimizer.
  Returns:
    An instance of ImageClassifier class.
  """
  if compat.get_tf_behavior() not in model_spec.compat_tf_versions:
    raise ValueError('Incompatible versions. Expect {}, but got {}.'.format(
        model_spec.compat_tf_versions, compat.get_tf_behavior()))

  # The hyperparameters for make_image_classifier by tensorflow hub.
  hparams = lib.get_default_hparams()
  if batch_size is not None:
    hparams = hparams._replace(batch_size=batch_size)
  if epochs is not None:
    hparams = hparams._replace(train_epochs=epochs)
  if train_whole_model is not None:
    hparams = hparams._replace(do_fine_tuning=train_whole_model)
  if dropout_rate is not None:
    hparams = hparams._replace(dropout_rate=dropout_rate)
  if learning_rate is not None:
    hparams = hparams._replace(learning_rate=learning_rate)
  if momentum is not None:
    hparams = hparams._replace(momentum=momentum)

  image_classifier = ImageClassifier(
      model_export_format,
      model_spec,
      train_data.index_to_label,
      train_data.num_classes,
      shuffle=shuffle,
      hparams=hparams)

  tf.compat.v1.logging.info('Retraining the models...')
  image_classifier.train(train_data, validation_data)

  return image_classifier


class ImageClassifier(classification_model.ClassificationModel):
  """ImageClassifier class for inference and exporting to tflite."""

  def __init__(self,
               model_export_format,
               model_spec,
               index_to_label,
               num_classes,
               shuffle=True,
               hparams=lib.get_default_hparams()):
    """Init function for ImageClassifier class.

    Args:
      model_export_format: Model export format such as saved_model / tflite.
      model_spec: Specification for the model.
      index_to_label: A list that map from index to label class name.
      num_classes: Number of label classes.
      shuffle: Whether the data should be shuffled.
      hparams: A namedtuple of hyperparameters. This function expects
        .dropout_rate: The fraction of the input units to drop, used in dropout
          layer.
    """
    super(ImageClassifier,
          self).__init__(model_export_format, model_spec, index_to_label,
                         num_classes, shuffle, hparams.do_fine_tuning)
    self.hparams = hparams
    self.model = self._create_model()

  def _create_model(self, hparams=None):
    """Creates the classifier model for retraining."""
    hparams = self._get_hparams_or_default(hparams)

    module_layer = hub_loader.HubKerasLayerV1V2(
        self.model_spec.uri, trainable=hparams.do_fine_tuning)
    return lib.build_model(module_layer, hparams,
                           self.model_spec.input_image_shape, self.num_classes)

  def train(self, train_data, validation_data=None, hparams=None):
    """Feeds the training data for training.

    Args:
      train_data: Training data.
      validation_data: Validation data. If None, skips validation process.
      hparams: A namedtuple of hyperparameters. This function expects
      .train_epochs: a Python integer with the number of passes over the
        training dataset;
      .learning_rate: a Python float forwarded to the optimizer;
      .momentum: a Python float forwarded to the optimizer;
      .batch_size: a Python integer, number of samples per training step.

    Returns:
      The tf.keras.callbacks.History object returned by tf.keras.Model.fit*().
    """
    hparams = self._get_hparams_or_default(hparams)

    train_ds = self._gen_dataset(
        train_data, hparams.batch_size, is_training=True)
    train_data_and_size = (train_ds, train_data.size)

    validation_ds = None
    validation_size = 0
    if validation_data is not None:
      validation_ds = self._gen_dataset(
          validation_data, hparams.batch_size, is_training=False)
      validation_size = validation_data.size
    validation_data_and_size = (validation_ds, validation_size)
    # Trains the models.
    return lib.train_model(self.model, hparams, train_data_and_size,
                           validation_data_and_size)

  def preprocess(self, image, label):
    """Image preprocessing method."""
    image = tf.cast(image, tf.float32)

    image -= tf.constant(
        self.model_spec.mean_rgb, shape=[1, 1, 3], dtype=image.dtype)
    image /= tf.constant(
        self.model_spec.stddev_rgb, shape=[1, 1, 3], dtype=image.dtype)

    image = tf.image.resize(image, self.model_spec.input_image_shape)
    label = tf.one_hot(label, depth=self.num_classes)
    return image, label

  def export(self,
             tflite_filename,
             label_filename,
             quantized=False,
             quantization_steps=None,
             representative_data=None):
    """Converts the retrained model based on `model_export_format`.

    Args:
      tflite_filename: File name to save tflite model.
      label_filename: File name to save labels.
      quantized: boolean, if True, save quantized model.
      quantization_steps: Number of post-training quantization calibration steps
        to run. Used only if `quantized` is True.
      representative_data: Representative data used for post-training
        quantization. Used only if `quantized` is True.
    """
    if self.model_export_format != mef.ModelExportFormat.TFLITE:
      raise ValueError('Model Export Format %s is not supported currently.' %
                       self.model_export_format)
    self._export_tflite(tflite_filename, label_filename, quantized,
                        quantization_steps, representative_data)

  def _get_hparams_or_default(self, hparams):
    """Returns hparams if not none, otherwise uses default one."""
    return hparams if hparams else self.hparams
