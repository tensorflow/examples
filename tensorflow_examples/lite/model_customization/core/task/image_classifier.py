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

import numpy as np
import tensorflow as tf # TF2

import tensorflow_examples.lite.model_customization.core.model_export_format as mef
from tensorflow_examples.lite.model_customization.core.task import classification_model
import tensorflow_examples.lite.model_customization.core.task.model_spec as ms

import tensorflow_hub as hub
from tensorflow_hub.tools.make_image_classifier import make_image_classifier_lib as lib


def create(data,
           model_export_format=mef.ModelExportFormat.TFLITE,
           model_spec=ms.mobilenet_v2_spec,
           shuffle=False,
           validation_ratio=0.1,
           test_ratio=0.1,
           batch_size=None,
           epochs=None,
           train_whole_model=None,
           dropout_rate=None,
           learning_rate=None,
           momentum=None):
  """Loads data and retrains the model based on data for image classification.

  Args:
    data: Raw data that could be splitted for training / validation / testing.
    model_export_format: Model export format such as saved_model / tflite.
    model_spec: Specification for the model.
    shuffle: Whether the data should be shuffled.
    validation_ratio: The ratio of valid data to be splitted.
    test_ratio: The ratio of test data to be splitted.
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
      data,
      model_export_format,
      model_spec,
      shuffle=shuffle,
      train_whole_model=False,
      validation_ratio=validation_ratio,
      test_ratio=test_ratio,
      hparams=hparams)

  tf.compat.v1.logging.info('Retraining the models...')
  image_classifier.train(hparams)

  return image_classifier


class ImageClassifier(classification_model.ClassificationModel):
  """ImageClassifier class for inference and exporting to tflite."""

  def __init__(self,
               data,
               model_export_format,
               model_spec,
               shuffle=True,
               train_whole_model=False,
               validation_ratio=0.1,
               test_ratio=0.1,
               hparams=lib.get_default_hparams()):
    """Init function for ImageClassifier class.

    Including splitting the raw input data into train/eval/test sets and
    selecting the exact NN model to be used.

    Args:
      data: Raw data that could be splitted for training / validation / testing.
      model_export_format: Model export format such as saved_model / tflite.
      model_spec: Specification for the model.
      shuffle: Whether the data should be shuffled.
      train_whole_model: If true, the Hub module is trained together with the
        classification layer on top. Otherwise, only train the top
        classification layer.
      validation_ratio: The ratio of valid data to be splitted.
      test_ratio: The ratio of test data to be splitted.
      hparams: A namedtuple of hyperparameters. This function expects
        .dropout_rate: The fraction of the input units to drop, used in dropout
          layer.
    """
    super(ImageClassifier,
          self).__init__(data, model_export_format, model_spec, shuffle,
                         train_whole_model, validation_ratio, test_ratio)

    # Gets pre_trained models.
    if model_export_format != mef.ModelExportFormat.TFLITE:
      raise ValueError('Model export mode %s is not supported currently.' %
                       str(model_export_format))
    self.pre_trained_model_spec = model_spec

    # Generates training, validation and testing data.
    if validation_ratio + test_ratio >= 1.0:
      raise ValueError(
          'The total ratio for validation and test data should be less than 1.0.'
      )

    self.valid_data, rest_data = data.split(validation_ratio, shuffle=shuffle)
    self.test_data, self.train_data = rest_data.split(
        test_ratio, shuffle=shuffle)

    # Checks dataset parameter.
    if self.train_data.size == 0:
      raise ValueError('Training dataset is empty.')

    # Creates the classifier model for retraining.
    module_layer = hub.KerasLayer(
        self.pre_trained_model_spec.uri, trainable=train_whole_model)
    self.model = lib.build_model(module_layer, hparams,
                                 self.pre_trained_model_spec.input_image_shape,
                                 data.num_classes)

  def _gen_train_dataset(self, data, batch_size=32):
    ds = data.dataset.map(self.preprocess_image)
    if self.shuffle:
      ds = ds.shuffle(buffer_size=self.train_data.size)
    ds = ds.repeat()
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds

  def _gen_valid_dataset(self, data, batch_size=32):
    ds = data.dataset.map(self.preprocess_image)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds

  def train(self, hparams=lib.get_default_hparams()):
    """Feeds the training data for training.

    Args:
      hparams: A namedtuple of hyperparameters. This function expects
      .train_epochs: a Python integer with the number of passes over the
        training dataset;
      .learning_rate: a Python float forwarded to the optimizer;
      .momentum: a Python float forwarded to the optimizer;
      .batch_size: a Python integer, number of samples per training step.

    Returns:
      The tf.keras.callbacks.History object returned by tf.keras.Model.fit*().
    """

    train_data_and_size = (self._gen_train_dataset(self.train_data,
                                                   hparams.batch_size),
                           self.train_data.size)
    validation_data_and_size = (self._gen_valid_dataset(self.valid_data,
                                                        hparams.batch_size),
                                self.valid_data.size)

    # Trains the models.
    return lib.train_model(self.model, hparams, train_data_and_size,
                           validation_data_and_size)

  def preprocess_image(self, image, label):
    """Image preprocessing method."""
    image = tf.cast(image, tf.float32)

    image -= tf.constant(
        self.pre_trained_model_spec.mean_rgb,
        shape=[1, 1, 3],
        dtype=image.dtype)
    image /= tf.constant(
        self.pre_trained_model_spec.stddev_rgb,
        shape=[1, 1, 3],
        dtype=image.dtype)

    image = tf.image.resize(image,
                            self.pre_trained_model_spec.input_image_shape)
    label = tf.one_hot(label, depth=self.data.num_classes)
    return image, label

  def evaluate(self, data=None, batch_size=32):
    """Evaluates the model.

    Args:
      data: Data to be evaluated. If None, then evaluates in self.test_data.
      batch_size: Number of samples per evaluation step.

    Returns:
      The loss value and accuracy.
    """
    if data is None:
      data = self.test_data
    ds = self._gen_valid_dataset(data, batch_size)

    return self.model.evaluate(ds)

  def export(self, tflite_filename, label_filename, **kwargs):
    """Converts the retrained model based on `model_export_format`.

    Args:
      tflite_filename: File name to save tflite model.
      label_filename: File name to save labels.
      **kwargs: Other parameters like `quantized` for TFLITE model.
    """
    if self.model_export_format == mef.ModelExportFormat.TFLITE:
      if 'quantized' in kwargs:
        quantized = kwargs['quantized']
      else:
        quantized = False
      self._export_tflite(tflite_filename, label_filename, quantized)
    else:
      raise ValueError('Model Export Format %s is not supported currently.' %
                       str(self.model_export_format))

  def _export_tflite(self, tflite_filename, label_filename, quantized):
    """Converts the retrained model to tflite format and saves it.

    Args:
      tflite_filename: File name to save tflite model.
      label_filename: File name to save labels.
      quantized: boolean, if True, save quantized model.
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
    if quantized:
      converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    tflite_model = converter.convert()

    with tf.io.gfile.GFile(tflite_filename, 'wb') as f:
      f.write(tflite_model)

    with tf.io.gfile.GFile(label_filename, 'w') as f:
      f.write('\n'.join(self.data.index_to_label))

    tf.compat.v1.logging.info('Export to tflite model %s, saved labels in %s.',
                              tflite_filename, label_filename)

  # TODO(b/142607208): need to fix the wrong output.
  def predict_topk(self, data=None, k=1, batch_size=32):
    """Predicts the top-k predictions.

    Args:
      data: Data to be evaluated. If None, then predicts in self.test_data.
      k: Number of top results to be predicted.
      batch_size: Number of samples per evaluation step.

    Returns:
      top k results. Each one is (label, probability).
    """
    if k < 0:
      raise ValueError('K should be equal or larger than 0.')

    if data is None:
      data = self.test_data
    ds = self._gen_valid_dataset(data, batch_size)

    predicted_prob = self.model.predict(ds)
    topk_prob, topk_id = tf.math.top_k(predicted_prob, k=k)
    topk_label = np.array(self.data.index_to_label)[topk_id]

    label_prob = []
    for label, prob in zip(topk_label, topk_prob.numpy()):
      label_prob.append(list(zip(label, prob)))

    return label_prob
