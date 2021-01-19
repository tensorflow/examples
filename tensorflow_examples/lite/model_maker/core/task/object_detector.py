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
"""ObjectDetector class."""

import tensorflow as tf
from tensorflow_examples.lite.model_maker.core import compat
from tensorflow_examples.lite.model_maker.core.task import custom_model
from tensorflow_examples.lite.model_maker.core.task import model_spec as ms


def create(train_data,
           model_spec,
           validation_data=None,
           epochs=None,
           batch_size=None,
           do_train=True):
  """Loads data and train the model for test classification.

  Args:
    train_data: Training data.
    model_spec: Specification for the model.
    validation_data: Validation data. If None, skips validation process.
    epochs: Number of epochs for training.
    batch_size: Batch size for training.
    do_train: Whether to run training.

  Returns:
    TextClassifier
  """
  model_spec = ms.get(model_spec)
  if compat.get_tf_behavior() not in model_spec.compat_tf_versions:
    raise ValueError('Incompatible versions. Expect {}, but got {}.'.format(
        model_spec.compat_tf_versions, compat.get_tf_behavior()))

  object_detector = ObjectDetector(model_spec, train_data.label_map)

  if do_train:
    tf.compat.v1.logging.info('Retraining the models...')
    object_detector.train(train_data, validation_data, epochs, batch_size)
  else:
    object_detector.create_model()

  return object_detector


class ObjectDetector(custom_model.CustomModel):
  """ObjectDetector class for inference and exporting to tflite."""

  def __init__(self, model_spec, label_map):
    super().__init__(model_spec, shuffle=None)
    if model_spec.config.label_map and model_spec.config.label_map != label_map:
      tf.compat.v1.logging.warn(
          'Label map is not the same as the previous label_map in model_spec.')
    model_spec.config.label_map = label_map
    model_spec.config.num_classes = len(label_map)

  def create_model(self):
    self.model = self.model_spec.create_model()
    return self.model

  def _get_dataset_and_steps(self, data, batch_size, is_training):
    """Gets dataset, steps and annotations json file."""
    if not data:
      return None, 0, None
    # TODO(b/171449557): Put this into DataLoader.
    dataset = data.gen_dataset(
        self.model_spec, batch_size, is_training=is_training)
    steps = len(data) // batch_size
    return dataset, steps, data.annotations_json_file

  def train(self,
            train_data,
            validation_data=None,
            epochs=None,
            batch_size=None):
    """Feeds the training data for training."""
    batch_size = batch_size if batch_size else self.model_spec.batch_size
    # TODO(b/171449557): Upstream this to the parent class.
    if len(train_data) < batch_size:
      raise ValueError('The size of the train_data (%d) couldn\'t be smaller '
                       'than batch_size (%d). To solve this problem, set '
                       'the batch_size smaller or increase the size of the '
                       'train_data.' % (len(train_data), batch_size))

    with self.model_spec.ds_strategy.scope():
      self.create_model()
      train_ds, steps_per_epoch, _ = self._get_dataset_and_steps(
          train_data, batch_size, is_training=True)
      validation_ds, validation_steps, val_json_file = self._get_dataset_and_steps(
          validation_data, batch_size, is_training=False)
      return self.model_spec.train(self.model, train_ds, steps_per_epoch,
                                   validation_ds, validation_steps, epochs,
                                   batch_size, val_json_file)

  def evaluate(self, data, batch_size=None):
    """Evaluates the model."""
    batch_size = batch_size if batch_size else self.model_spec.batch_size
    ds = data.gen_dataset(self.model_spec, batch_size, is_training=False)
    steps = len(data) // batch_size
    # TODO(b/171449557): Upstream this to the parent class.
    if steps <= 0:
      raise ValueError('The size of the validation_data (%d) couldn\'t be '
                       'smaller than batch_size (%d). To solve this problem, '
                       'set the batch_size smaller or increase the size of the '
                       'validation_data.' % (len(data), batch_size))

    return self.model_spec.evaluate(self.model, ds, steps,
                                    data.annotations_json_file)

  def _export_saved_model(self, saved_model_dir):
    """Saves the model to Tensorflow SavedModel."""
    self.model_spec.export_saved_model(saved_model_dir)

  def _export_tflite(self, tflite_filepath, quantization_config=None):
    """Converts the retrained model to tflite format and saves it.

    Args:
      tflite_filepath: File path to save tflite model.
      quantization_config: Configuration for post-training quantization.
    """
    self.model_spec.export_tflite(tflite_filepath, quantization_config)
