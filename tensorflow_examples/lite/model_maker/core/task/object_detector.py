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
from typing import Dict, Optional, Tuple, TypeVar

import tensorflow as tf
from tensorflow_examples.lite.model_maker.core import compat
from tensorflow_examples.lite.model_maker.core.data_util import object_detector_dataloader
from tensorflow_examples.lite.model_maker.core.export_format import ExportFormat
from tensorflow_examples.lite.model_maker.core.export_format import QuantizationType
from tensorflow_examples.lite.model_maker.core.task import configs
from tensorflow_examples.lite.model_maker.core.task import custom_model
from tensorflow_examples.lite.model_maker.core.task import model_spec as ms
from tensorflow_examples.lite.model_maker.core.task.metadata_writers.object_detector import metadata_writer_for_object_detector as metadata_writer
from tensorflow_examples.lite.model_maker.core.task.model_spec import object_detector_spec

from tensorflow_examples.lite.model_maker.third_party.efficientdet.keras import label_util

T = TypeVar('T', bound='ObjectDetector')


def create(train_data: object_detector_dataloader.DataLoader,
           model_spec: object_detector_spec.EfficientDetModelSpec,
           validation_data: Optional[
               object_detector_dataloader.DataLoader] = None,
           epochs: Optional[object_detector_dataloader.DataLoader] = None,
           batch_size: Optional[int] = None,
           train_whole_model: bool = False,
           do_train: bool = True) -> T:
  """Loads data and train the model for object detection.

  Args:
    train_data: Training data.
    model_spec: Specification for the model.
    validation_data: Validation data. If None, skips validation process.
    epochs: Number of epochs for training.
    batch_size: Batch size for training.
    train_whole_model: Boolean, False by default. If true, train the whole
      model. Otherwise, only train the layers that are not match
      `model_spec.config.var_freeze_expr`.
    do_train: Whether to run training.

  Returns:
    ObjectDetector
  """
  model_spec = ms.get(model_spec)
  if train_whole_model:
    model_spec.config.var_freeze_expr = None
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


def _get_model_info(
    model_spec: object_detector_spec.EfficientDetModelSpec,
    quantization_type: Optional[QuantizationType] = None,
    quantization_config: Optional[configs.QuantizationConfig] = None,
) -> metadata_writer.ModelSpecificInfo:
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
  elif quantization_type == QuantizationType.INT8:
    image_min = 0
    image_max = 255

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

  def __init__(self, model_spec: object_detector_spec.EfficientDetModelSpec,
               label_map: Dict[int, str]) -> None:
    super().__init__(model_spec, shuffle=None)
    if model_spec.config.label_map and model_spec.config.label_map != label_map:
      tf.compat.v1.logging.warn(
          'Label map is not the same as the previous label_map in model_spec.')
    model_spec.config.label_map = label_map
    # TODO(yuqili): num_classes = 1 have some issues during training. Thus we
    # make minimum num_classes=2 for now.
    model_spec.config.num_classes = max(2, max(label_map.keys()))

  def create_model(self) -> tf.keras.Model:
    self.model = self.model_spec.create_model()
    return self.model

  def _get_dataset_and_steps(
      self,
      data: object_detector_dataloader.DataLoader,
      batch_size: int,
      is_training: bool,
  ) -> Tuple[Optional[tf.data.Dataset], int, Optional[str]]:
    """Gets dataset, steps and annotations json file."""
    if not data:
      return None, 0, None
    # TODO(b/171449557): Put this into DataLoader.
    dataset = data.gen_dataset(
        self.model_spec, batch_size, is_training=is_training)
    steps = len(data) // batch_size
    return dataset, steps, data.annotations_json_file

  def train(self,
            train_data: object_detector_dataloader.DataLoader,
            validation_data: Optional[
                object_detector_dataloader.DataLoader] = None,
            epochs: Optional[int] = None,
            batch_size: Optional[int] = None) -> tf.keras.Model:
    """Feeds the training data for training."""
    if not self.model_spec.config.drop_remainder:
      raise ValueError('Must set `drop_remainder=True` during training. '
                       'Otherwise it will fail.')

    batch_size = batch_size if batch_size else self.model_spec.batch_size
    # TODO(b/171449557): Upstream this to the parent class.
    if len(train_data) < batch_size:
      raise ValueError('The size of the train_data (%d) couldn\'t be smaller '
                       'than batch_size (%d). To solve this problem, set '
                       'the batch_size smaller or increase the size of the '
                       'train_data.' % (len(train_data), batch_size))
    if validation_data and len(validation_data) < batch_size:
      tf.compat.v1.logging.warn(
          'The size of the validation_data (%d) is smaller than batch_size '
          '(%d). Ignore the validation_data.' %
          (len(validation_data), batch_size))
      validation_data = None

    with self.model_spec.ds_strategy.scope():
      self.create_model()
      train_ds, steps_per_epoch, _ = self._get_dataset_and_steps(
          train_data, batch_size, is_training=True)
      validation_ds, validation_steps, val_json_file = self._get_dataset_and_steps(
          validation_data, batch_size, is_training=False)
      return self.model_spec.train(self.model, train_ds, steps_per_epoch,
                                   validation_ds, validation_steps, epochs,
                                   batch_size, val_json_file)

  def evaluate(self,
               data: object_detector_dataloader.DataLoader,
               batch_size: Optional[int] = None) -> Dict[str, float]:
    """Evaluates the model."""
    batch_size = batch_size if batch_size else self.model_spec.batch_size
    # Not to drop the smaller batch to evaluate the whole dataset.
    self.model_spec.config.drop_remainder = False
    ds = data.gen_dataset(self.model_spec, batch_size, is_training=False)
    steps = (len(data) + batch_size - 1) // batch_size
    # TODO(b/171449557): Upstream this to the parent class.
    if steps <= 0:
      raise ValueError('The size of the validation_data (%d) couldn\'t be '
                       'smaller than batch_size (%d). To solve this problem, '
                       'set the batch_size smaller or increase the size of the '
                       'validation_data.' % (len(data), batch_size))

    eval_metrics = self.model_spec.evaluate(self.model, ds, steps,
                                            data.annotations_json_file)
    # Set back drop_remainder=True since it must be True during training.
    # Otherwise it will fail.
    self.model_spec.config.drop_remainder = True
    return eval_metrics

  def evaluate_tflite(
      self, tflite_filepath: str,
      data: object_detector_dataloader.DataLoader) -> Dict[str, float]:
    """Evaluate the TFLite model."""
    ds = data.gen_dataset(self.model_spec, batch_size=1, is_training=False)
    return self.model_spec.evaluate_tflite(tflite_filepath, ds, len(data),
                                           data.annotations_json_file)

  def _export_saved_model(self, saved_model_dir: str) -> None:
    """Saves the model to Tensorflow SavedModel."""
    self.model_spec.export_saved_model(saved_model_dir)

  def _export_tflite(
      self,
      tflite_filepath: str,
      quantization_type: QuantizationType = QuantizationType.INT8,
      representative_data: Optional[
          object_detector_dataloader.DataLoader] = None,
      quantization_config: Optional[configs.QuantizationConfig] = None,
      with_metadata: bool = True,
      export_metadata_json_file: bool = False) -> None:
    """Converts the retrained model to tflite format and saves it.

    Args:
      tflite_filepath: File path to save tflite model.
      quantization_type: Enum, type of post-training quantization. Accepted
        values are `INT8`, `FP16`, `FP32`, `DYNAMIC`. `FP16` means float16
        quantization with 2x smaller, optimized for GPU. `INT8` means full
        integer quantization with 4x smaller, 3x+ speedup, optimized for Edge
        TPU. 'DYNAMIC' means dynamic range quantization with	4x smaller, 2x-3x
        speedup. `FP32` mean exporting float model without quantization. Please
        refer to
        https://www.tensorflow.org/lite/performance/post_training_quantization
        for more detailed about different techniques for post-training
        quantization.
      representative_data: Representative dataset for full integer
        quantization. Used when `quantization_type=INT8`.
      quantization_config: Configuration for post-training quantization.
      with_metadata: Whether the output tflite model contains metadata.
      export_metadata_json_file: Whether to export metadata in json file. If
        True, export the metadata in the same directory as tflite model.Used
        only if `with_metadata` is True.
    """
    if quantization_type and quantization_config:
      raise ValueError('At most one of the paramaters `quantization_type` and '
                       '`quantization_config` can be set.')
    if quantization_type == QuantizationType.INT8 and \
       representative_data is None:
      raise ValueError('`representative_data` must be set when '
                       '`quantization_type=QuantizationType.INT8.')

    ds, _, _ = self._get_dataset_and_steps(
        representative_data, batch_size=1, is_training=False)

    self.model_spec.export_tflite(tflite_filepath, quantization_type, ds,
                                  quantization_config)

    if with_metadata:
      with tempfile.TemporaryDirectory() as temp_dir:
        tf.compat.v1.logging.info(
            'Label file is inside the TFLite model with metadata.')
        label_filepath = os.path.join(temp_dir, 'labelmap.txt')
        self._export_labels(label_filepath)
        model_info = _get_model_info(self.model_spec, quantization_type,
                                     quantization_config)
        export_dir = os.path.dirname(tflite_filepath)
        populator = metadata_writer.MetadataPopulatorForObjectDetector(
            tflite_filepath, export_dir, model_info, label_filepath)
        populator.populate(export_metadata_json_file)

  def _export_labels(self, label_filepath: str) -> None:
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
