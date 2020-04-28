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

import os

import tensorflow.compat.v2 as tf

from tensorflow_examples.lite.model_maker.core import compat
from tensorflow_examples.lite.model_maker.core.export_format import ExportFormat
from tensorflow_examples.lite.model_maker.core.task import classification_model
from tensorflow_examples.lite.model_maker.core.task import model_spec as ms


def create(train_data,
           model_spec=ms.AverageWordVecModelSpec(),
           shuffle=False,
           batch_size=32,
           epochs=None,
           validation_data=None):
  """Loads data and train the model for test classification.

  Args:
    train_data: Training data.
    model_spec: Specification for the model.
    shuffle: Whether the data should be shuffled.
    batch_size: Batch size for training.
    epochs: Number of epochs for training.
    validation_data: Validation data. If None, skips validation process.

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

  tf.compat.v1.logging.info('Retraining the models...')
  text_classifier.train(train_data, validation_data, epochs, batch_size)

  return text_classifier


class TextClassifier(classification_model.ClassificationModel):
  """TextClassifier class for inference and exporting to tflite."""

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

  def get_dataset_fn(self, input_data, global_batch_size, is_training):
    """Gets a closure to create a dataset."""

    def _dataset_fn(ctx=None):
      """Returns tf.data.Dataset for text classifier retraining."""
      batch_size = ctx.get_per_replica_batch_size(
          global_batch_size) if ctx else global_batch_size
      dataset = self._gen_dataset(
          input_data,
          batch_size,
          is_training=is_training,
          input_pipeline_context=ctx)
      return dataset

    return _dataset_fn

  def train(self, train_data, validation_data=None, epochs=None, batch_size=32):
    """Feeds the training data for training."""

    train_input_fn = self.get_dataset_fn(
        train_data, batch_size, is_training=True)

    validation_steps = 0
    validation_input_fn = None
    if validation_data is not None:
      validation_input_fn = self.get_dataset_fn(
          validation_data, batch_size, is_training=False)
      validation_steps = validation_data.size // batch_size

    steps_per_epoch = train_data.size // batch_size

    self.model = self.model_spec.run_classifier(train_input_fn,
                                                validation_input_fn, epochs,
                                                steps_per_epoch,
                                                validation_steps,
                                                self.num_classes)

    return self.model

  @staticmethod
  def _set_batch_size(model, batch_size):
    """Sets batch size for the model."""
    for model_input in model.inputs:
      new_shape = [batch_size] + model_input.shape[1:]
      model_input.set_shape(new_shape)

  def export(self,
             export_dir,
             tflite_filename='model.tflite',
             label_filename='labels.txt',
             vocab_filename='vocab',
             saved_model_filename='saved_model',
             export_format=None,
             **kwargs):
    """Converts the retrained model based on `export_format`.

    Args:
      export_dir: The directory to save exported files.
      tflite_filename: File name to save tflite model. The full export path is
        {export_dir}/{tflite_filename}.
      label_filename: File name to save labels. The full export path is
        {export_dir}/{label_filename}.
      vocab_filename: File name to save vocabulary.  The full export path is
        {export_dir}/{vocab_filename}.
      saved_model_filename: Path to SavedModel or H5 file to save the model. The
        full export path is
        {export_dir}/{saved_model_filename}/{saved_model.pb|assets|variables}.
      export_format: List of export format that could be saved_model, tflite,
        label, vocab.
      **kwargs: Other parameters like `quantized` for TFLITE model.
    """
    # Default export ExportFormat are TFLite models and labels.
    if export_format is None:
      export_format = [
          ExportFormat.TFLITE, ExportFormat.LABEL, ExportFormat.VOCAB
      ]
    if not isinstance(export_format, list):
      export_format = [export_format]

    if not tf.io.gfile.exists(export_dir):
      tf.io.gfile.makedirs(export_dir)

    if ExportFormat.LABEL in export_format:
      label_filepath = os.path.join(export_dir, label_filename)
      self._export_labels(label_filepath)

    if ExportFormat.TFLITE in export_format:
      tflite_filepath = os.path.join(export_dir, tflite_filename)
      self._export_tflite(tflite_filepath, **kwargs)

    if ExportFormat.SAVED_MODEL in export_format:
      saved_model_filepath = os.path.join(export_dir, saved_model_filename)
      self._export_saved_model(saved_model_filepath, **kwargs)

    if ExportFormat.VOCAB in export_format:
      vocab_filepath = os.path.join(export_dir, vocab_filename)
      self.model_spec.save_vocab(vocab_filepath)

  def _export_tflite(self,
                     tflite_filepath,
                     quantized=False,
                     quantization_steps=None,
                     representative_data=None):
    """Converts the retrained model to tflite format and saves it.

    Args:
      tflite_filepath: File path to save tflite model.
      quantized: boolean, if True, save quantized model.
      quantization_steps: Number of post-training quantization calibration steps
        to run. Used only if `quantized` is True.
      representative_data: Representative data used for post-training
        quantization. Used only if `quantized` is True.
    """
    # Sets batch size from None to 1 when converting to tflite.
    self._set_batch_size(self.model, batch_size=1)
    super(TextClassifier,
          self)._export_tflite(tflite_filepath, quantized, quantization_steps,
                               representative_data)
    # Sets batch size back to None to support retraining later.
    self._set_batch_size(self.model, batch_size=None)
