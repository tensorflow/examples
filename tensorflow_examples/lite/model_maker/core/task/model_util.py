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
