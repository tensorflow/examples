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
"""SGD optimizer implementation for transfer learning models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.compat.v1 as tfv1


class SGD(object):
  """SGD optimizer configuration for transfer learning converter."""

  def __init__(self, learning_rate):
    self._learning_rate = learning_rate

  def generate_optimizer_model(self, parameter_shapes):
    """Generates a TFLite model that represents an optimizer step.

    The generated model accepts as inputs model parameters' current
    values and gradients, and returns as outputs the new values.

    Args:
      parameter_shapes: list of model parameter shapes.

    Returns:
      TFLite optimizer model.
    """
    with tfv1.Session(graph=tf.Graph()) as sess:
      current_values = [
          tfv1.placeholder(tf.float32, shape) for shape in parameter_shapes
      ]
      gradients = [
          tfv1.placeholder(tf.float32, shape) for shape in parameter_shapes
      ]

      new_values = [
          current - self._learning_rate * gradient
          for current, gradient in zip(current_values, gradients)
      ]
      converter = tfv1.lite.TFLiteConverter.from_session(
          sess, current_values + gradients, new_values)
      return converter.convert()
