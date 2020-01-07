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
"""Custom model that is already retained by data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import numpy as np
import tensorflow as tf # TF2
import tensorflow_examples.lite.model_customization.core.model_export_format as mef


class ClassificationModel(abc.ABC):
  """"The abstract base class that represents a Tensorflow classification model."""

  def __init__(self, data, model_export_format, model_spec, shuffle,
               train_whole_model, validation_ratio, test_ratio):
    """Initialize a instance with data, deploy mode and other related parameters.

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
    """
    if model_export_format != mef.ModelExportFormat.TFLITE:
      raise ValueError('Model export format %s is not supported currently.' %
                       str(model_export_format))

    self.data = data
    self.model_export_format = model_export_format
    self.model_spec = model_spec
    self.shuffle = shuffle
    self.train_whole_model = train_whole_model
    self.validation_ratio = validation_ratio
    self.test_ratio = test_ratio

    # Generates training, validation and testing data.
    if validation_ratio + test_ratio >= 1.0:
      raise ValueError(
          'The total ratio for validation and test data should be less than 1.0.'
      )

    self.validation_data, rest_data = data.split(
        validation_ratio, shuffle=shuffle)
    self.test_data, self.train_data = rest_data.split(
        test_ratio / (1 - validation_ratio), shuffle=shuffle)

    # Checks dataset parameter.
    if self.train_data.size == 0:
      raise ValueError('Training dataset is empty.')

    self.model = None

  @abc.abstractmethod
  def _create_model(self, **kwargs):
    return

  @abc.abstractmethod
  def preprocess(self, sample_data, label):
    return

  @abc.abstractmethod
  def train(self, **kwargs):
    return

  @abc.abstractmethod
  def export(self, **kwargs):
    return

  def summary(self):
    self.model.summary()

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
    ds = self._gen_validation_dataset(data, batch_size)

    return self.model.evaluate(ds)

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
    ds = self._gen_validation_dataset(data, batch_size)

    predicted_prob = self.model.predict(ds)
    topk_prob, topk_id = tf.math.top_k(predicted_prob, k=k)
    topk_label = np.array(self.data.index_to_label)[topk_id.numpy()]

    label_prob = []
    for label, prob in zip(topk_label, topk_prob.numpy()):
      label_prob.append(list(zip(label, prob)))

    return label_prob

  def _gen_train_dataset(self, data, batch_size=32):
    """Generates training dataset."""
    ds = data.dataset.map(
        self.preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if self.shuffle:
      ds = ds.shuffle(buffer_size=data.size)
    ds = ds.repeat()
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds

  def _gen_validation_dataset(self, data, batch_size=32):
    """Generates validation dataset."""
    ds = data.dataset.map(
        self.preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds

  def _export_tflite(self, tflite_filename, label_filename, quantized=False):
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
