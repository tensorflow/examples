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
import os

import tensorflow as tf

from tensorflow_examples.lite.model_maker.core.export_format import ExportFormat


class CustomModel(abc.ABC):
  """"The abstract base class that represents a Tensorflow classification model."""

  DEFAULT_EXPORT_FORMAT = []
  ALLOWED_EXPORT_FORMAT = []

  def __init__(self, model_spec, shuffle):
    """Initialize a instance with data, deploy mode and other related parameters.

    Args:
      model_spec: Specification for the model.
      shuffle: Whether the data should be shuffled.
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
      dataset = self._gen_dataset(
          input_data,
          batch_size,
          is_training=is_training,
          input_pipeline_context=ctx)
      return dataset

    return _dataset_fn

  def _get_input_fn_and_steps(self, data, batch_size, is_training):
    """Gets input_fn and steps for training/evaluation."""
    if data is None:
      input_fn = None
      steps = 0
    else:
      input_fn = self._get_dataset_fn(data, batch_size, is_training)
      steps = data.size // batch_size
    return input_fn, steps

  def _gen_dataset(self,
                   data,
                   batch_size=32,
                   is_training=True,
                   input_pipeline_context=None):
    """Generates training / validation dataset."""
    # The dataset is always sharded by number of hosts.
    # num_input_pipelines is the number of hosts rather than number of cores.
    ds = data.dataset
    if input_pipeline_context and input_pipeline_context.num_input_pipelines > 1:
      ds = ds.shard(input_pipeline_context.num_input_pipelines,
                    input_pipeline_context.input_pipeline_id)

    ds = ds.map(
        self.preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if is_training:
      if self.shuffle:
        ds = ds.shuffle(buffer_size=min(data.size, 100))
      ds = ds.repeat()

    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds

  def _get_export_format(self, export_format):
    if export_format is None:
      export_format = self.DEFAULT_EXPORT_FORMAT

    if not isinstance(export_format, list):
      export_format = [export_format]

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
    export_format = self._get_export_format(export_format)

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
