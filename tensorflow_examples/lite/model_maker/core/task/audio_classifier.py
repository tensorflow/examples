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
"""APIs to train an audio classification model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_examples.lite.model_maker.core.api import mm_export
from tensorflow_examples.lite.model_maker.core.export_format import ExportFormat
from tensorflow_examples.lite.model_maker.core.task import classification_model
from tensorflow_examples.lite.model_maker.core.task import model_util
from tensorflow_examples.lite.model_maker.core.task.model_spec import audio_spec


@mm_export('audio_classifier.AudioClassifier')
class AudioClassifier(classification_model.ClassificationModel):
  """Audio classifier for training/inference and exporing."""

  # TODO(b/171848856): Add TFJS export.
  DEFAULT_EXPORT_FORMAT = (ExportFormat.LABEL, ExportFormat.TFLITE)
  ALLOWED_EXPORT_FORMAT = (ExportFormat.LABEL, ExportFormat.TFLITE,
                           ExportFormat.SAVED_MODEL)

  def _get_dataset_and_steps(self, data, batch_size, is_training):
    if not data:
      return None, 0
    # TODO(b/171449557): Put this into DataLoader.
    input_fn, steps = self._get_input_fn_and_steps(
        data, batch_size, is_training=is_training)
    dataset = tf.distribute.get_strategy().distribute_datasets_from_function(
        input_fn)
    return dataset, steps

  def train(self, train_data, validation_data, epochs, batch_size):
    # TODO(b/171449557): Upstream this to the parent class.
    if len(train_data) < batch_size:
      raise ValueError('The size of the train_data (%d) couldn\'t be smaller '
                       'than batch_size (%d). To solve this problem, set '
                       'the batch_size smaller or increase the size of the '
                       'train_data.' % (len(train_data), batch_size))

    with self.model_spec.strategy.scope():
      train_ds, _ = self._get_dataset_and_steps(
          train_data, batch_size, is_training=True)
      validation_ds, _ = self._get_dataset_and_steps(
          validation_data, batch_size, is_training=False)

      self.model = self.model_spec.create_model(
          train_data.num_classes, train_whole_model=self.train_whole_model)

      # Display model summary
      self.model.summary()

      return self.model_spec.run_classifier(
          self.model,
          epochs,
          train_ds,
          validation_ds,
          callbacks=self._keras_callbacks(self.model_spec.model_dir))

  def _export_tflite(self, tflite_filepath, quantization_config='default'):
    """Converts the retrained model to tflite format and saves it.

    Args:
      tflite_filepath: File path to save tflite model.
      quantization_config: Configuration for post-training quantization.
    """
    if quantization_config == 'default':
      quantization_config = self.model_spec.get_default_quantization_config()

    # Allow model_spec to override this method.
    fn = getattr(self.model_spec, 'export_tflite', None)
    if not callable(fn):
      fn = model_util.export_tflite
    fn(self.model, tflite_filepath, quantization_config)

  def confusion_matrix(self, data, batch_size=32):
    # TODO(b/171449557): Consider moving this to ClassificationModel
    ds = data.gen_dataset(
        batch_size, is_training=False, preprocess=self.preprocess)
    predicated = []
    truth = []
    for item, label in ds:
      if tf.rank(label) == 2:  # One-hot encoded labels (batch, num_classes)
        truth.extend(tf.math.argmax(label, axis=-1))
        predicated.extend(tf.math.argmax(self.model.predict(item), axis=-1))
      else:
        truth.extend(label)
        predicated.extend(self.model.predict(item))

    return tf.math.confusion_matrix(
        labels=truth, predictions=predicated, num_classes=data.num_classes)

  @classmethod
  def create(cls,
             train_data,
             model_spec,
             validation_data=None,
             batch_size=32,
             epochs=5,
             model_dir=None,
             do_train=True,
             train_whole_model=False):
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
      train_whole_model: Boolean. By default, only the classification head is
        trained. When True, the base model is also trained.

    Returns:
      An instance based on AudioClassifier.
    """
    if not isinstance(model_spec, audio_spec.BaseSpec):
      model_spec = model_spec.get(model_spec, model_dir=model_dir)
    task = cls(
        model_spec,
        train_data.index_to_label,
        shuffle=True,
        train_whole_model=train_whole_model)
    if do_train:
      task.train(train_data, validation_data, epochs, batch_size)
    return task


# Shortcut function.
create = AudioClassifier.create
mm_export('audio_classifier.create').export_constant(__name__, 'create')
