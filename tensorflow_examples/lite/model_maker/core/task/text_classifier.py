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
"""TextClassier class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_examples.lite.model_maker.core import compat
from tensorflow_examples.lite.model_maker.core.export_format import ExportFormat
from tensorflow_examples.lite.model_maker.core.task import classification_model
from tensorflow_examples.lite.model_maker.core.task import model_spec as ms
from tensorflow_examples.lite.model_maker.core.task import model_util


def create(train_data,
           model_spec=ms.AverageWordVecModelSpec(),
           validation_data=None,
           batch_size=None,
           epochs=3,
           shuffle=False,
           do_train=True):
  """Loads data and train the model for test classification.

  Args:
    train_data: Training data.
    model_spec: Specification for the model.
    validation_data: Validation data. If None, skips validation process.
    batch_size: Batch size for training.
    epochs: Number of epochs for training.
    shuffle: Whether the data should be shuffled.
    do_train: Whether to run training.

  Returns:
    TextClassifier
  """
  if compat.get_tf_behavior() not in model_spec.compat_tf_versions:
    raise ValueError('Incompatible versions. Expect {}, but got {}.'.format(
        model_spec.compat_tf_versions, compat.get_tf_behavior()))

  text_classifier = TextClassifier(
      model_spec,
      train_data.index_to_label,
      train_data.num_classes,
      shuffle=shuffle)

  if do_train:
    tf.compat.v1.logging.info('Retraining the models...')
    text_classifier.train(train_data, validation_data, epochs, batch_size)
  else:
    text_classifier.create_model()

  return text_classifier


class TextClassifier(classification_model.ClassificationModel):
  """TextClassifier class for inference and exporting to tflite."""

  DEFAULT_EXPORT_FORMAT = [
      ExportFormat.TFLITE, ExportFormat.LABEL, ExportFormat.VOCAB
  ]
  ALLOWED_EXPORT_FORMAT = [
      ExportFormat.TFLITE, ExportFormat.LABEL, ExportFormat.VOCAB,
      ExportFormat.SAVED_MODEL
  ]

  def __init__(self,
               model_spec,
               index_to_label,
               num_classes,
               shuffle=True):
    """Init function for TextClassifier class.

    Args:
      model_spec: Specification for the model.
      index_to_label: A list that map from index to label class name.
      num_classes: Number of label classes.
      shuffle: Whether the data should be shuffled.
    """
    super(TextClassifier, self).__init__(
        model_spec,
        index_to_label,
        num_classes,
        shuffle,
        train_whole_model=True)

  def create_model(self):
    self.model = self.model_spec.create_model(self.num_classes)

  def train(self,
            train_data,
            validation_data=None,
            epochs=None,
            batch_size=None):
    """Feeds the training data for training."""
    if batch_size is None:
      batch_size = self.model_spec.default_batch_size

    if train_data.size < batch_size:
      raise ValueError('The size of the train_data (%d) couldn\'t be smaller '
                       'than batch_size (%d). To solve this problem, set '
                       'the batch_size smaller or increase the size of the '
                       'train_data.' % (train_data.size, batch_size))

    train_input_fn, steps_per_epoch = self._get_input_fn_and_steps(
        train_data, batch_size, is_training=True)
    validation_input_fn, validation_steps = self._get_input_fn_and_steps(
        validation_data, batch_size, is_training=False)

    self.model = self.model_spec.run_classifier(train_input_fn,
                                                validation_input_fn, epochs,
                                                steps_per_epoch,
                                                validation_steps,
                                                self.num_classes)

    return self.model

  def _export_tflite(self, tflite_filepath, quantization_config=None):
    """Converts the retrained model to tflite format and saves it.

    Args:
      tflite_filepath: File path to save tflite model.
      quantization_config: Configuration for post-training quantization.
    """
    # Sets batch size from None to 1 when converting to tflite.
    model_util.set_batch_size(self.model, batch_size=1)
    model_util.export_tflite(self.model, tflite_filepath, quantization_config,
                             self._gen_dataset,
                             self.model_spec.convert_from_saved_model_tf2)
    # Sets batch size back to None to support retraining later.
    model_util.set_batch_size(self.model, batch_size=None)
