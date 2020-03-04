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
"""Compat modules."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf # TF2

_DEFAULT_TF_BEHAVIOR = 2

# Get version of tf behavior in use (valid 1 or 2).
_tf_behavior_version = _DEFAULT_TF_BEHAVIOR


def setup_tf_behavior(tf_version=_DEFAULT_TF_BEHAVIOR):
  """Setup tf behavior. It must be used before the main()."""
  global _tf_behavior_version
  if tf_version not in [1, 2]:
    raise ValueError(
        'tf_version should be in [1, 2], but got {}'.format(tf_version))

  if tf_version == 1:
    tf.compat.v1.logging.warn(
        'Using v1 behavior. Please note that it is mainly to run legacy models,'
        'however v2 is more preferrable if they are supported.')
    tf.compat.v1.disable_v2_behavior()
  else:
    assert tf.__version__.startswith('2')
  _tf_behavior_version = tf_version


def get_tf_behavior():
  """Gets version for tf behavior.

  Returns:
    int, 1 or 2 indicating the behavior version.
  """
  return _tf_behavior_version

