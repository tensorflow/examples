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
"""Base custom model that is already retained by data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import inspect
import os

import tensorflow as tf

from tensorflow_examples.lite.model_maker.core.export_format import ExportFormat
from tensorflow_examples.lite.model_maker.core.task import model_util


def _get_params(f, **kwargs):
  """Gets parameters of the function `f` from `**kwargs`."""
  parameters = inspect.signature(f).parameters
  f_kwargs = {}  # kwargs for the function `f`
  for param_name in parameters.keys():
    if param_name in kwargs:
      f_kwargs[param_name] = kwargs.pop(param_name)
  return f_kwargs, kwargs


class CustomModel(abc.ABC):
  """"The abstract base class that represents a Tensorflow classification model."""

  DEFAULT_EXPORT_FORMAT = (ExportFormat.TFLITE)
  ALLOWED_EXPORT_FORMAT = (ExportFormat.TFLITE, ExportFormat.SAVED_MODEL)

  def __init__(self, model_spec, shuffle):
    """Initialize a instance with data, deploy mode and other related parameters.

    Args:
      model_spec: Specification for the model.
      shuffle: Whether the training data should be shuffled.
    """
    self.model_spec = model_spec
    self.shuffle = shuffle
    self.model = None

  def preprocess(self, sample_data, label):
    """Preprocess the data."""
    # TODO(yuqili): remove this method once preprocess for image classifier is
    # also moved to DataLoader part.
    return sample_data, label

  @abc.abstractmethod
  def train(self, train_data, validation_data=None, **kwargs):
    return

  def summary(self):
    self.model.summary()

  @abc.abstractmethod
  def evaluate(self, data, **kwargs):
    return

  # TODO(b/155949323): Refactor the code for gen_dataset in CustomModel to a
  # seperated  dataloader.
  def _get_dataset_fn(self, input_data, global_batch_size, is_training):
    """Gets a closure to create a dataset."""

    def _dataset_fn(ctx=None):
      """Returns tf.data.Dataset for question answer retraining."""
      batch_size = ctx.get_per_replica_batch_size(
          global_batch_size) if ctx else global_batch_size
      dataset = input_data.gen_dataset(
          batch_size,
          is_training=is_training,
          # TODO(wangtz): Consider moving `shuffle` to DataLoader.
          shuffle=self.shuffle,
          input_pipeline_context=ctx,
          preprocess=self.preprocess)
      return dataset

    return _dataset_fn

  def _get_input_fn_and_steps(self, data, batch_size, is_training):
    """Gets input_fn and steps for training/evaluation."""
    if data is None:
      input_fn = None
      steps = 0
    else:
      input_fn = self._get_dataset_fn(data, batch_size, is_training)
      steps = len(data) // batch_size
    return input_fn, steps

  def _get_default_export_format(self, **kwargs):
    """Gets the default export format."""
    if kwargs.get('with_metadata', True):
      export_format = (ExportFormat.TFLITE)
    else:
      export_format = self.DEFAULT_EXPORT_FORMAT
    return export_format

  def _get_export_format(self, export_format, **kwargs):
    """Get export format."""
    if export_format is None:
      export_format = self._get_default_export_format(**kwargs)

    if not isinstance(export_format, (list, tuple)):
      export_format = (export_format,)

    # Checks whether each export format is allowed.
    for e_format in export_format:
      if e_format not in self.ALLOWED_EXPORT_FORMAT:
        raise ValueError('Export format %s is not allowed.' % e_format)

    return export_format

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
    export_format = self._get_export_format(export_format, **kwargs)

    if not tf.io.gfile.exists(export_dir):
      tf.io.gfile.makedirs(export_dir)

    if ExportFormat.TFLITE in export_format:
      with_metadata = kwargs.get('with_metadata', True)
      tflite_filepath = os.path.join(export_dir, tflite_filename)
      export_tflite_kwargs, kwargs = _get_params(self._export_tflite, **kwargs)
      self._export_tflite(tflite_filepath, **export_tflite_kwargs)
    else:
      with_metadata = False

    if ExportFormat.SAVED_MODEL in export_format:
      saved_model_filepath = os.path.join(export_dir, saved_model_filename)
      export_saved_model_kwargs, kwargs = _get_params(self._export_saved_model,
                                                      **kwargs)
      self._export_saved_model(saved_model_filepath,
                               **export_saved_model_kwargs)

    if ExportFormat.VOCAB in export_format:
      if with_metadata:
        tf.compat.v1.logging.warn('Export a separated vocab file even though '
                                  'vocab file is already inside the TFLite '
                                  'model with metadata.')
      vocab_filepath = os.path.join(export_dir, vocab_filename)
      self.model_spec.save_vocab(vocab_filepath)

    if ExportFormat.LABEL in export_format:
      if with_metadata:
        tf.compat.v1.logging.warn('Export a separated label file even though '
                                  'label file is already inside the TFLite '
                                  'model with metadata.')
      label_filepath = os.path.join(export_dir, label_filename)
      self._export_labels(label_filepath)

    if kwargs:
      tf.compat.v1.logging.warn('Encountered unknown parameters: ' +
                                str(kwargs))

  def _export_saved_model(self,
                          filepath,
                          overwrite=True,
                          include_optimizer=True,
                          save_format=None,
                          signatures=None,
                          options=None):
    """Saves the model to Tensorflow SavedModel or a single HDF5 file.

    Args:
      filepath: String, path to SavedModel or H5 file to save the model.
      overwrite: Whether to silently overwrite any existing file at the target
        location, or provide the user with a manual prompt.
      include_optimizer: If True, save optimizer's state together.
      save_format: Either 'tf' or 'h5', indicating whether to save the model to
        Tensorflow SavedModel or HDF5. Defaults to 'tf' in TF 2.X, and 'h5' in
        TF 1.X.
      signatures: Signatures to save with the SavedModel. Applicable to the 'tf'
        format only. Please see the `signatures` argument in
        `tf.saved_model.save` for details.
      options: Optional `tf.saved_model.SaveOptions` object that specifies
        options for saving to SavedModel.
    """
    if filepath is None:
      raise ValueError(
          "SavedModel filepath couldn't be None when exporting to SavedModel.")
    self.model.save(filepath, overwrite, include_optimizer, save_format,
                    signatures, options)

  def _export_tflite(self, tflite_filepath, quantization_config=None):
    """Converts the retrained model to tflite format and saves it.

    Args:
      tflite_filepath: File path to save tflite model.
      quantization_config: Configuration for post-training quantization.
    """
    model_util.export_tflite(self.model, tflite_filepath, quantization_config)

  def _keras_callbacks(self, model_dir):
    """Returns a list of default keras callbacks for `model.fit`."""
    summary_dir = os.path.join(model_dir, 'summaries')
    summary_callback = tf.keras.callbacks.TensorBoard(summary_dir)
    checkpoint_path = os.path.join(model_dir, 'checkpoint')
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path, save_weights_only=True)
    return [summary_callback, checkpoint_callback]
