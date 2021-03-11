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
"""APIs to train an object detection model."""

import os
import tempfile
from typing import Dict

import tensorflow as tf
from tensorflow_examples.lite.model_maker.core import compat
from tensorflow_examples.lite.model_maker.core.data_util import object_detector_dataloader
from tensorflow_examples.lite.model_maker.core.export_format import ExportFormat
from tensorflow_examples.lite.model_maker.core.task import custom_model
from tensorflow_examples.lite.model_maker.core.task import model_spec as ms
from tensorflow_examples.lite.model_maker.core.task.metadata_writers.object_detector import metadata_writer_for_object_detector as metadata_writer

from tensorflow_examples.lite.model_maker.third_party.efficientdet.keras import label_util


def create(train_data,
           model_spec,
           validation_data=None,
           epochs=None,
           batch_size=None,
           do_train=True):
  """Loads data and train the model for object detection.

  Args:
    train_data: Training data.
    model_spec: Specification for the model.
    validation_data: Validation data. If None, skips validation process.
    epochs: Number of epochs for training.
    batch_size: Batch size for training.
    do_train: Whether to run training.

  Returns:
    ObjectDetector
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


def _get_model_info(model_spec, quantization_config=None):
  """Gets the specific info for the object detection model."""

  # Gets image_min/image_max for float/quantized model.
  image_min = -1
  image_max = 1
  if quantization_config:
    if quantization_config.inference_input_type == tf.uint8:
      image_min = 0
      image_max = 255
    elif quantization_config.inference_input_type == tf.int8:
      image_min = -128
      image_max = 127

  def _get_list(v):
    if isinstance(v, list) or isinstance(v, tuple):
      return v
    else:
      return [v]

  return metadata_writer.ModelSpecificInfo(
      name=model_spec.model_name,
      version='v1',
      image_width=model_spec.config.image_size[1],
      image_height=model_spec.config.image_size[0],
      image_min=image_min,
      image_max=image_max,
      mean=_get_list(model_spec.config.mean_rgb),
      std=_get_list(model_spec.config.stddev_rgb))


class ObjectDetector(custom_model.CustomModel):
  """ObjectDetector class for inference and exporting to tflite."""

  ALLOWED_EXPORT_FORMAT = (ExportFormat.TFLITE, ExportFormat.SAVED_MODEL,
                           ExportFormat.LABEL)

  def __init__(self, model_spec, label_map):
    super().__init__(model_spec, shuffle=None)
    if model_spec.config.label_map and model_spec.config.label_map != label_map:
      tf.compat.v1.logging.warn(
          'Label map is not the same as the previous label_map in model_spec.')
    model_spec.config.label_map = label_map
    # TODO(yuqili): num_classes = 1 have some issues during training. Thus we
    # make minimum num_classes=2 for now.
    model_spec.config.num_classes = max(2, max(label_map.keys()))

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

  def evaluate_tflite(
      self, tflite_filepath: str,
      data: object_detector_dataloader.DataLoader) -> Dict[str, float]:
    """Evaluate the TFLite model."""
    ds = data.gen_dataset(self.model_spec, batch_size=1, is_training=False)
    return self.model_spec.evaluate_tflite(tflite_filepath, ds, len(data),
                                           data.annotations_json_file)

  def _export_saved_model(self, saved_model_dir):
    """Saves the model to Tensorflow SavedModel."""
    self.model_spec.export_saved_model(saved_model_dir)

  def _export_tflite(self,
                     tflite_filepath,
                     quantization_config=None,
                     with_metadata=True,
                     export_metadata_json_file=False):
    """Converts the retrained model to tflite format and saves it.

    Args:
      tflite_filepath: File path to save tflite model.
      quantization_config: Configuration for post-training quantization.
      with_metadata: Whether the output tflite model contains metadata.
      export_metadata_json_file: Whether to export metadata in json file. If
        True, export the metadata in the same directory as tflite model.Used
        only if `with_metadata` is True.
    """
    self.model_spec.export_tflite(tflite_filepath, quantization_config)

    if with_metadata:
      with tempfile.TemporaryDirectory() as temp_dir:
        tf.compat.v1.logging.info(
            'Label file is inside the TFLite model with metadata.')
        label_filepath = os.path.join(temp_dir, 'labelmap.txt')
        self._export_labels(label_filepath)
        model_info = _get_model_info(self.model_spec, quantization_config)
        export_dir = os.path.dirname(tflite_filepath)
        populator = metadata_writer.MetadataPopulatorForObjectDetector(
            tflite_filepath, export_dir, model_info, label_filepath)
        populator.populate(export_metadata_json_file)

  def _export_labels(self, label_filepath):
    """Export labels to label_filepath."""
    tf.compat.v1.logging.info('Saving labels in %s.', label_filepath)
    num_classes = self.model_spec.config.num_classes
    label_map = label_util.get_label_map(self.model_spec.config.label_map)
    with tf.io.gfile.GFile(label_filepath, 'w') as f:
      # Ignores label_map[0] that's the background. The labels in the label file
      # for TFLite metadata should start from the actual labels without the
      # background.
      for i in range(num_classes):
        label = label_map[i + 1] if i + 1 in label_map else '???'
        f.write(label + '\n')
