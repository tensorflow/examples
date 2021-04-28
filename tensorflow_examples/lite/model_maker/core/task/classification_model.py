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
"""Custom classification model that is already retained by data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow_examples.lite.model_maker.core.data_util import data_util
from tensorflow_examples.lite.model_maker.core.export_format import ExportFormat
from tensorflow_examples.lite.model_maker.core.task import custom_model
from tensorflow_examples.lite.model_maker.core.task import model_util


class ClassificationModel(custom_model.CustomModel):
  """"The abstract base class that represents a Tensorflow classification model."""

  DEFAULT_EXPORT_FORMAT = (ExportFormat.TFLITE, ExportFormat.LABEL)
  ALLOWED_EXPORT_FORMAT = (ExportFormat.TFLITE, ExportFormat.LABEL,
                           ExportFormat.SAVED_MODEL, ExportFormat.TFJS)

  def __init__(self, model_spec, index_to_label, shuffle, train_whole_model):
    """Initialize a instance with data, deploy mode and other related parameters.

    Args:
      model_spec: Specification for the model.
      index_to_label: A list that map from index to label class name.
      shuffle: Whether the data should be shuffled.
      train_whole_model: If true, the Hub module is trained together with the
        classification layer on top. Otherwise, only train the top
        classification layer.
    """
    super(ClassificationModel, self).__init__(model_spec, shuffle)
    self.index_to_label = index_to_label
    self.num_classes = len(index_to_label)
    self.train_whole_model = train_whole_model

  def evaluate(self, data, batch_size=32):
    """Evaluates the model.

    Args:
      data: Data to be evaluated.
      batch_size: Number of samples per evaluation step.

    Returns:
      The loss value and accuracy.
    """
    ds = data.gen_dataset(
        batch_size, is_training=False, preprocess=self.preprocess)
    return self.model.evaluate(ds)

  def predict_top_k(self, data, k=1, batch_size=32):
    """Predicts the top-k predictions.

    Args:
      data: Data to be evaluated.
      k: Number of top results to be predicted.
      batch_size: Number of samples per evaluation step.

    Returns:
      top k results. Each one is (label, probability).
    """
    if k < 0:
      raise ValueError('K should be equal or larger than 0.')
    ds = data.gen_dataset(
        batch_size, is_training=False, preprocess=self.preprocess)

    predicted_prob = self.model.predict(ds)
    topk_prob, topk_id = tf.math.top_k(predicted_prob, k=k)
    topk_label = np.array(self.index_to_label)[topk_id.numpy()]

    label_prob = []
    for label, prob in zip(topk_label, topk_prob.numpy()):
      label_prob.append(list(zip(label, prob)))

    return label_prob

  def _export_labels(self, label_filepath):
    if label_filepath is None:
      raise ValueError("Label filepath couldn't be None when exporting labels.")

    tf.compat.v1.logging.info('Saving labels in %s.', label_filepath)
    with tf.io.gfile.GFile(label_filepath, 'w') as f:
      f.write('\n'.join(self.index_to_label))

  def evaluate_tflite(self, tflite_filepath, data, postprocess_fn=None):
    """Evaluates the tflite model.

    Args:
      tflite_filepath: File path to the TFLite model.
      data: Data to be evaluated.
      postprocess_fn: Postprocessing function that will be applied to the output
        of `lite_runner.run` before calculating the probabilities.

    Returns:
      The evaluation result of TFLite model - accuracy.
    """
    ds = data.gen_dataset(
        batch_size=1, is_training=False, preprocess=self.preprocess)

    predictions, labels = [], []

    lite_runner = model_util.get_lite_runner(tflite_filepath, self.model_spec)
    for i, (feature, label) in enumerate(data_util.generate_elements(ds)):
      log_steps = 1000
      tf.compat.v1.logging.log_every_n(tf.compat.v1.logging.DEBUG,
                                       'Processing example: #%d\n%s', log_steps,
                                       i, feature)

      probabilities = lite_runner.run(feature)

      if postprocess_fn:
        probabilities = postprocess_fn(probabilities)
      predictions.append(np.argmax(probabilities))

      # Gets the ground-truth labels.
      label = label[0]
      if label.size > 1:  # one-hot tensor.
        label = np.argmax(label)
      labels.append(label)

    predictions, labels = np.array(predictions), np.array(labels)
    result = {'accuracy': (predictions == labels).mean()}
    return result
