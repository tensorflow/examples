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
"""Utilities for keras models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import tempfile
from typing import Optional

import numpy as np
import tensorflow as tf

from tensorflow_examples.lite.model_maker.core import compat
from tensorflow_examples.lite.model_maker.core.export_format import QuantizationType
from tensorflow_examples.lite.model_maker.core.task import configs
from tensorflowjs.converters import converter as tfjs_converter
from tflite_support import metadata as _metadata

DEFAULT_SCALE, DEFAULT_ZERO_POINT = 0, 0
_NUM_CALIBRATION_STEPS = 500


def set_batch_size(model, batch_size):
  """Sets batch size for the model."""
  for model_input in model.inputs:
    new_shape = [batch_size] + model_input.shape[1:]
    model_input.set_shape(new_shape)


def _create_temp_dir(convert_from_saved_model):
  """Creates temp dir, if True is given."""
  if convert_from_saved_model:
    return tempfile.TemporaryDirectory()
  else:
    return DummyContextManager()


class DummyContextManager(object):

  def __enter__(self):
    pass

  def __exit__(self, *args):
    pass


def export_tflite(model,
                  tflite_filepath,
                  quantization_config=None,
                  convert_from_saved_model_tf2=False,
                  preprocess=None,
                  supported_ops=(tf.lite.OpsSet.TFLITE_BUILTINS,),
                  quantization_type: Optional[QuantizationType] = None,
                  representative_dataset: Optional[tf.data.Dataset] = None):
  """Converts the retrained model to tflite format and saves it.

  Args:
    model: model to be converted to tflite.
    tflite_filepath: File path to save tflite model.
    quantization_config: Configuration for post-training quantization.
    convert_from_saved_model_tf2: Convert to TFLite from saved_model in TF 2.x.
    preprocess: A preprocess function to apply on the dataset.
        # TODO(wangtz): Remove when preprocess is split off from CustomModel.
    supported_ops: A list of supported ops in the converted TFLite file.
    quantization_type: Enum, type of post-training quantization. Accepted values
      are `INT8`, `FP16`, `FP32`, `DYNAMIC`. `FP16` means float16 quantization
      with 2x smaller, optimized for GPU. `INT8` means full integer quantization
      with 4x smaller, 3x+ speedup, optimized for EdgeTPU. 'DYNAMIC' means
      dynamic range quantization with	4x smaller, 2x-3x speedup. `FP32` mean
      exporting float model without quantization. Please refer to
      https://www.tensorflow.org/lite/performance/post_training_quantization
      for more detailed about different techniques for post-training
      quantization.
    representative_dataset: Representative dataset for full integer
      quantization. Used when `quantization_type=INT8`.
  """
  if quantization_type and quantization_config:
    raise ValueError('At most one of the paramaters `quantization_type` and '
                     '`quantization_config` can be set.')

  if tflite_filepath is None:
    raise ValueError(
        "TFLite filepath couldn't be None when exporting to tflite.")

  if compat.get_tf_behavior() == 1:
    lite = tf.compat.v1.lite
  else:
    lite = tf.lite

  convert_from_saved_model = (
      compat.get_tf_behavior() == 1 or convert_from_saved_model_tf2)
  with _create_temp_dir(convert_from_saved_model) as temp_dir_name:
    if temp_dir_name:
      save_path = os.path.join(temp_dir_name, 'saved_model')
      model.save(save_path, include_optimizer=False, save_format='tf')
      converter = lite.TFLiteConverter.from_saved_model(save_path)
    else:
      converter = lite.TFLiteConverter.from_keras_model(model)

    # Uses `quantization_type` for post-training quantization.
    if quantization_type == QuantizationType.INT8:
      if representative_dataset is None:
        raise ValueError('`representative_data` must be set when '
                         '`quantization_type=QuantizationType.INT8.')

      converter.representative_dataset = configs.get_representative_dataset_gen(
          representative_dataset, _NUM_CALIBRATION_STEPS)
      converter.optimizations = [tf.lite.Optimize.DEFAULT]
      converter.inference_input_type = tf.uint8
      converter.inference_output_type = tf.uint8
      supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
      converter.target_spec.supported_ops = supported_ops
    elif quantization_type == QuantizationType.FP16:
      converter.optimizations = [tf.lite.Optimize.DEFAULT]
      converter.target_spec.supported_types = [tf.float16]
    elif quantization_type == QuantizationType.DYNAMIC:
      converter.optimizations = [tf.lite.Optimize.DEFAULT]
    elif quantization_type and quantization_type != QuantizationType.FP32:
      raise ValueError('Unsupported `quantization_type`: %s' %
                       str(quantization_type))
    # Uses `quantization_config` for post-training quantization.
    elif quantization_config:
      converter = quantization_config.get_converter_with_quantization(
          converter, preprocess=preprocess)

    converter.target_spec.supported_ops = supported_ops
    tflite_model = converter.convert()

  with tf.io.gfile.GFile(tflite_filepath, 'wb') as f:
    f.write(tflite_model)


def get_lite_runner(tflite_filepath, model_spec=None):
  """Gets `LiteRunner` from file path to TFLite model and `model_spec`."""
  # Gets the functions to handle the input & output indexes if exists.
  reorder_input_details_fn = None
  if hasattr(model_spec, 'reorder_input_details'):
    reorder_input_details_fn = model_spec.reorder_input_details

  reorder_output_details_fn = None
  if hasattr(model_spec, 'reorder_output_details'):
    reorder_output_details_fn = model_spec.reorder_output_details

  lite_runner = LiteRunner(tflite_filepath, reorder_input_details_fn,
                           reorder_output_details_fn)
  return lite_runner


def _get_input_tensor(input_tensors, input_details, i):
  """Gets input tensor in `input_tensors` that maps `input_detail[i]`."""
  if isinstance(input_tensors, dict):
    # Gets the mapped input tensor.
    input_detail = input_details[i]
    for input_tensor_name, input_tensor in input_tensors.items():
      if input_tensor_name in input_detail['name']:
        return input_tensor
    raise ValueError('Input tensors don\'t contains a tensor that mapped the '
                     'input detail %s' % str(input_detail))
  else:
    return input_tensors[i]


class LiteRunner(object):
  """Runs inference with the TFLite model."""

  def __init__(self,
               tflite_filepath,
               reorder_input_details_fn=None,
               reorder_output_details_fn=None):
    """Initializes Lite runner with tflite model file.

    Args:
      tflite_filepath: File path to the TFLite model.
      reorder_input_details_fn: Function to reorder the input details to map the
        order of keras model.
      reorder_output_details_fn: Function to reorder the output details to map
        the order of keras model.
    """
    with tf.io.gfile.GFile(tflite_filepath, 'rb') as f:
      tflite_model = f.read()
    self.interpreter = tf.lite.Interpreter(model_content=tflite_model)
    self.interpreter.allocate_tensors()

    # Gets the indexed of the input tensors.
    self.input_details = self.interpreter.get_input_details()
    if reorder_input_details_fn is not None:
      self.input_details = reorder_input_details_fn(self.input_details)

    self.output_details = self.interpreter.get_output_details()
    if reorder_output_details_fn is not None:
      self.output_details = reorder_output_details_fn(self.output_details)

  def run(self, input_tensors):
    """Runs inference with the TFLite model.

    Args:
      input_tensors: List / Dict of the input tensors of the TFLite model. The
        order should be the same as the keras model if it's a list. It also
        accepts tensor directly if the model has only 1 input.

    Returns:
      List of the output tensors for multi-output models, otherwise just
        the output tensor. The order should be the same as the keras model.
    """

    if not isinstance(input_tensors, list) and \
       not isinstance(input_tensors, tuple) and \
       not isinstance(input_tensors, dict):
      input_tensors = [input_tensors]

    interpreter = self.interpreter

    # Reshape inputs
    for i, input_detail in enumerate(self.input_details):
      input_tensor = _get_input_tensor(input_tensors, self.input_details, i)
      interpreter.resize_tensor_input(input_detail['index'], input_tensor.shape)
    interpreter.allocate_tensors()

    # Feed input to the interpreter
    for i, input_detail in enumerate(self.input_details):
      input_tensor = _get_input_tensor(input_tensors, self.input_details, i)
      if input_detail['quantization'] != (DEFAULT_SCALE, DEFAULT_ZERO_POINT):
        # Quantize the input
        scale, zero_point = input_detail['quantization']
        input_tensor = input_tensor / scale + zero_point
        input_tensor = np.array(input_tensor, dtype=input_detail['dtype'])
      interpreter.set_tensor(input_detail['index'], input_tensor)

    interpreter.invoke()

    output_tensors = []
    for output_detail in self.output_details:
      output_tensor = interpreter.get_tensor(output_detail['index'])
      if output_detail['quantization'] != (DEFAULT_SCALE, DEFAULT_ZERO_POINT):
        # Dequantize the output
        scale, zero_point = output_detail['quantization']
        output_tensor = output_tensor.astype(np.float32)
        output_tensor = (output_tensor - zero_point) * scale
      output_tensors.append(output_tensor)

    if len(output_tensors) == 1:
      return output_tensors[0]
    return output_tensors


def export_tfjs(keras_or_saved_model,
                output_dir,
                tflite_filepath=None,
                **kwargs):
  """Exports saved model to tfjs.

  https://www.tensorflow.org/js/guide/conversion?hl=en

  Args:
    keras_or_saved_model: Keras or saved model.
    output_dir: Output TF.js model dir.
    tflite_filepath: str, file path to existing tflite model. If set, the
      metadata is extracted to the TF.js model.
    **kwargs: Other options.
  """
  # For Keras model, creates a saved model first in a temp dir. Otherwise,
  # convert directly.
  is_keras = isinstance(keras_or_saved_model, tf.keras.Model)
  with _create_temp_dir(is_keras) as temp_dir_name:
    # Export keras model to saved model and then convert to TFJS.
    if is_keras:
      keras_or_saved_model.save(
          temp_dir_name, include_optimizer=False, save_format='tf')
      path = temp_dir_name
    else:
      path = keras_or_saved_model

    # Extract metadata if tflite_filepath is provided.
    if tflite_filepath:
      metadata_json = extract_tflite_metadata_json(tflite_filepath)
      metadata = json.loads(metadata_json)
      kwargs.update(metadata=metadata)

    tfjs_converter.dispatch_keras_saved_model_to_tensorflowjs_conversion(
        path, output_dir, **kwargs)


def load_tfjs_keras_model(model_path):
  """Loads tfjs keras model from path."""
  return tfjs_converter.keras_tfjs_loader.load_keras_model(
      model_path, load_weights=True)


def extract_tflite_metadata_json(tflite_filepath):
  """Extracts metadata from tflite model filepath.

  Args:
    tflite_filepath: str, path to tflite model file.

  Returns:
    str: tflite metadata json string.
  """
  displayer = _metadata.MetadataDisplayer.with_model_file(tflite_filepath)
  return displayer.get_metadata_json()
