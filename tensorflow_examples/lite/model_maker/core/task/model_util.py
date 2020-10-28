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

import os
import tempfile

import numpy as np
import tensorflow as tf
from tensorflow_examples.lite.model_maker.core import compat


def set_batch_size(model, batch_size):
  """Sets batch size for the model."""
  for model_input in model.inputs:
    new_shape = [batch_size] + model_input.shape[1:]
    model_input.set_shape(new_shape)


def _create_temp_dir(convert_from_saved_model):
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
                  gen_dataset_fn=None,
                  convert_from_saved_model_tf2=False):
  """Converts the retrained model to tflite format and saves it.

  Args:
    model: model to be converted to tflite.
    tflite_filepath: File path to save tflite model.
    quantization_config: Configuration for post-training quantization.
    gen_dataset_fn: Function to generate tf.data.dataset from
      `representative_data`. Used only when `representative_data` in
      `quantization_config` is setted.
    convert_from_saved_model_tf2: Convert to TFLite from saved_model in TF 2.x.
  """
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

    if quantization_config:
      converter = quantization_config.get_converter_with_quantization(
          converter, gen_dataset_fn)

    tflite_model = converter.convert()

  with tf.io.gfile.GFile(tflite_filepath, 'wb') as f:
    f.write(tflite_model)


def get_lite_runner(tflite_filepath, model_spec):
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
    for i, input_detail in enumerate(self.input_details):
      input_tensor = _get_input_tensor(input_tensors, self.input_details, i)

      default_scale, default_zero_point = 0, 0
      if input_detail['quantization'] != (default_scale, default_zero_point):
        # Quantize the input
        scale, zero_point = input_detail['quantization']
        input_tensor = input_tensor / scale + zero_point
        input_tensor = np.array(input_tensor, dtype=input_detail['dtype'])
      interpreter.set_tensor(input_detail['index'], input_tensor)

    interpreter.invoke()

    output_tensors = []
    for output_detail in self.output_details:
      output_tensor = interpreter.get_tensor(output_detail['index'])
      if output_detail['quantization'] != (0, 0):
        # Dequantize the output
        scale, zero_point = output_detail['quantization']
        output_tensor = output_tensor.astype(np.float32)
        output_tensor = (output_tensor - zero_point) * scale
      output_tensors.append(output_tensor)

    if len(output_tensors) == 1:
      return output_tensors[0]
    return output_tensors
