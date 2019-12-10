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
"""Adam optimizer implementation for transfer learning models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.compat.v1 as tfv1


class Adam(object):
  """Adam optimizer configuration for transfer learning converter."""

  def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
    self._learning_rate = learning_rate
    self._beta1 = beta1
    self._beta2 = beta2
    self._eps = eps

  def generate_optimizer_model(self, parameter_shapes):
    """Generates a TFLite model that represents an optimizer step.

    The generated model inputs are current values of the trainable
    model parameters, followed by their gradients, and then by
    the current mutable optimizer state.

    The generated model outputs are the new values of the trainable
    parameters, followed by the updated mutable optimizer state.

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
      ms = [tfv1.placeholder(tf.float32, shape) for shape in parameter_shapes]
      vs = [tfv1.placeholder(tf.float32, shape) for shape in parameter_shapes]
      step = tfv1.placeholder(tf.float32, ())

      new_values = []
      new_ms = []
      new_vs = []
      for cur_param, grad, m, v in zip(current_values, gradients, ms, vs):
        m = (1 - self._beta1) * grad + self._beta1 * m
        v = (1 - self._beta2) * (grad**2) + self._beta2 * v
        mhat = m / (1 - self._beta1**(step + 1))
        vhat = v / (1 - self._beta2**(step + 1))
        new_param = cur_param - (
            self._learning_rate * mhat / (tf.sqrt(vhat) + self._eps))
        new_values.append(new_param)
        new_ms.append(m)
        new_vs.append(v)

      converter = tfv1.lite.TFLiteConverter.from_session(
          sess, current_values + gradients + ms + vs + [step],
          new_values + new_ms + new_vs + [step + 1])
      return converter.convert()
