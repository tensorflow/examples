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
"""Test util for model maker."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
import shutil
import tempfile

from absl import flags
import numpy as np
import PIL.Image

import tensorflow.compat.v2 as tf
from tensorflow_examples.lite.model_maker.core import compat
from tensorflow_examples.lite.model_maker.core.data_util import dataloader
from tensorflow_examples.lite.model_maker.core.task import model_util

FLAGS = flags.FLAGS


def test_srcdir():
  """Returns the path where to look for test data files."""
  if "test_srcdir" in flags.FLAGS:
    return flags.FLAGS["test_srcdir"].value
  elif "TEST_SRCDIR" in os.environ:
    return os.environ["TEST_SRCDIR"]
  else:
    raise RuntimeError("Missing TEST_SRCDIR environment.")


def get_test_data_path(file_or_dirname):
  """Return full test data path."""
  for (directory, subdirs, files) in tf.io.gfile.walk(test_srcdir()):
    for f in subdirs + files:
      if f.endswith(file_or_dirname):
        return os.path.join(directory, f)
  raise ValueError("No %s in test directory" % file_or_dirname)


def get_cache_dir(temp_dir, filename):
  """Gets `cache_dir` for `tf.keras.utils.get_file` function."""
  # Copies SST-2.zip in testdata folder to a temp folder since
  # `tf.keras.utils.get_file` needed writability of the path.
  try:
    src_path = get_test_data_path(filename)
    dest_path = os.path.join(temp_dir, "datasets")
    if not tf.io.gfile.exists(dest_path):
      tf.io.gfile.mkdir(dest_path)
    shutil.copy2(src_path, dest_path)
    return temp_dir
  except ValueError:  # There's no testdata.
    return None


def test_in_tf_1(fn):
  """Decorator to test in tf 1 behaviors."""

  @functools.wraps(fn)
  def decorator(*args, **kwargs):
    if compat.get_tf_behavior() != 1:
      tf.compat.v1.logging.info("Skip function {} for test_in_tf_1".format(
          fn.__name__))
      return
    fn(*args, **kwargs)

  return decorator


def test_in_tf_2(fn):
  """Decorator to test in tf 2 behaviors."""

  @functools.wraps(fn)
  def decorator(*args, **kwargs):
    if compat.get_tf_behavior() != 2:
      tf.compat.v1.logging.info("Skip function {} for test_in_tf_2".format(
          fn.__name__))
      return
    fn(*args, **kwargs)

  return decorator


def test_in_tf_1and2(fn):
  """Decorator to test in tf 1 and 2 behaviors."""

  @functools.wraps(fn)
  def decorator(*args, **kwargs):
    if compat.get_tf_behavior() not in [1, 2]:
      tf.compat.v1.logging.info("Skip function {} for test_in_tf_1and2".format(
          fn.__name__))
      return
    fn(*args, **kwargs)

  return decorator


def build_model(input_shape, num_classes):
  """Builds a simple model for test."""
  inputs = tf.keras.layers.Input(shape=input_shape)
  if len(input_shape) == 3:  # Image inputs.
    outputs = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(outputs)
  elif len(input_shape) == 1:  # Text inputs.
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(inputs)
  else:
    raise ValueError("Model inputs should be 2D tensor or 4D tensor.")

  model = tf.keras.Model(inputs=inputs, outputs=outputs)
  return model


def get_dataloader(data_size, input_shape, num_classes, max_input_value=1000):
  """Gets a simple `DataLoader` object for test."""
  features = tf.random.uniform(
      shape=[data_size] + input_shape,
      minval=0,
      maxval=max_input_value,
      dtype=tf.float32)

  labels = tf.random.uniform(
      shape=[data_size], minval=0, maxval=num_classes, dtype=tf.int32)

  ds = tf.data.Dataset.from_tensor_slices((features, labels))
  data = dataloader.DataLoader(ds, data_size)
  return data


def create_pascal_voc(temp_dir=None):
  """Creates test data with PASCAL VOC format."""
  if temp_dir is None or not tf.io.gfile.exists(temp_dir):
    temp_dir = tempfile.mkdtemp()

  # Saves the image into images_dir.
  image_file_name = "2012_12.jpg"
  image_data = np.random.rand(256, 256, 3)
  images_dir = os.path.join(temp_dir, "images")
  os.mkdir(images_dir)
  save_path = os.path.join(images_dir, image_file_name)
  image = PIL.Image.fromarray(image_data, "RGB")
  image.save(save_path)

  # Gets the annonation path.
  annotations_path = get_test_data_path("2012_12.xml")
  annotations_dir = os.path.dirname(annotations_path)

  label_map = {
      1: "person",
      2: "notperson",
  }
  return images_dir, annotations_dir, label_map


def is_same_output(tflite_file,
                   keras_model,
                   input_tensors,
                   model_spec=None,
                   atol=1e-04):
  """Whether the output of TFLite model is the same as keras model."""
  # Gets output from lite model.
  lite_runner = model_util.get_lite_runner(tflite_file, model_spec)
  lite_output = lite_runner.run(input_tensors)

  # Gets output from keras model.
  keras_output = keras_model.predict_on_batch(input_tensors)

  return np.allclose(lite_output, keras_output, atol=atol)
