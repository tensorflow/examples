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
"""AudioClassifier class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_examples.lite.model_maker.core.export_format import ExportFormat
from tensorflow_examples.lite.model_maker.core.task import classification_model
from tensorflow_examples.lite.model_maker.core.task.model_spec import audio_spec


def create(train_data,
           model_spec,
           validation_data=None,
           batch_size=256,
           epochs=1,
           model_dir=None,
           do_train=True):
  """Loads data and retrains the model.

  Args:
    train_data: A instance of audio_dataloader.DataLoader class.
    model_spec: Specification for the model.
    validation_data: Validation DataLoader. If None, skips validation process.
    batch_size: Number of samples per training step. If `use_hub_library` is
      False, it represents the base learning rate when train batch size is 256
      and it's linear to the batch size.
    epochs: Number of epochs for training.
    model_dir: The location of the model checkpoint files.
    do_train: Whether to run training.

  Returns:
    An instance of AudioClassifier class.
  """
  if not isinstance(model_spec, audio_spec.BaseSpec):
    model_spec = model_spec.get(model_spec, model_dir=model_dir)
  task = AudioClassifier(
      model_spec,
      train_data.index_to_label,
      shuffle=True,
      train_whole_model=False)
  if do_train:
    task.train(train_data, validation_data, epochs, batch_size)
  return task


class AudioClassifier(classification_model.ClassificationModel):
  """Audio classifier for training/inference and exporing."""

  # TODO(b/171848856): Add TFLite/TFJS export.
  DEFAULT_EXPORT_FORMAT = (ExportFormat.LABEL, ExportFormat.SAVED_MODEL)
  ALLOWED_EXPORT_FORMAT = (ExportFormat.LABEL, ExportFormat.SAVED_MODEL)

  def _get_dataset_and_steps(self, data, batch_size, is_training):
    if not data:
      return None, 0
    # TODO(b/171449557): Put this into DataLoader.
    input_fn, steps = self._get_input_fn_and_steps(
        data, batch_size, is_training=is_training)
    dataset = tf.distribute.get_strategy(
    ).experimental_distribute_datasets_from_function(input_fn)
    return dataset, steps

  def train(self, train_data, validation_data, epochs, batch_size):
    # TODO(b/171449557): Upstream this to the parent class.
    if len(train_data) < batch_size:
      raise ValueError('The size of the train_data (%d) couldn\'t be smaller '
                       'than batch_size (%d). To solve this problem, set '
                       'the batch_size smaller or increase the size of the '
                       'train_data.' % (len(train_data), batch_size))

    with self.model_spec.strategy.scope():
      train_ds, train_steps = self._get_dataset_and_steps(
          train_data, batch_size, is_training=True)
      validation_ds, validation_steps = self._get_dataset_and_steps(
          validation_data, batch_size, is_training=False)

      self.model = self.model_spec.create_model(train_data.num_classes)
      return self.model_spec.run_classifier(
          self.model,
          epochs,
          train_ds,
          train_steps,
          validation_ds,
          validation_steps,
          callbacks=self._keras_callbacks(self.model_spec.model_dir))
