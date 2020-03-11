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

from absl import flags

import tensorflow as tf # TF2
from tensorflow_examples.lite.model_maker.core import compat

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
  for directory, subdirs, files in tf.io.gfile.walk(test_srcdir()):
    for f in subdirs + files:
      if f.endswith(file_or_dirname):
        return os.path.join(directory, f)
  raise ValueError("No %s in test directory" % file_or_dirname)


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
