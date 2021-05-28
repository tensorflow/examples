# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""
Custom Gradient example illustrated using Polynomial Gradient.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class PolynomialGradient(tf.Module):
  """PolynomialGradient to compute gradients of polynomial.

  This initializes itself with `weights` (trainable variable) and `order`
  (highest power of polynomial).

  The module provides method `calc_gradient` which is wrapped inside
  `tf.custom_gradient` and will compute gradient of resultant `poly` w.r.t.
  `x` as well as `weights`.
  """
  def __init__(self, weights, order):
    self.weights = weights
    self.order = order

  @tf.custom_gradient
  def calc_gradient(self, x):
    # Creating a polynomial by the conventional formula.
    # If order is 2, then it will create
    # poly = weights[-3]* x ** 2 + weights[-2] * x + weights[-1]
    poly = self.weights[-1]
    for k in range(self.order):
      poly += x ** (k+1) * self.weights[-(k+2)]

    def grad_fn(dpoly, variables=None):
      # Computing gradient of poly w.r.t. x
      # After computing, we need to left multiply it with dpoly
      grad_xs = self.weights[-2]
      for k in range(self.order-1):
        coefficient = self.weights[-(k+3)]
        raw_exponent = k + 2
        grad_xs += coefficient * raw_exponent * x ** (raw_exponent - 1)
      grad_xs = dpoly * grad_xs

      # Computing gradient w.r.t. trainable variable i.e. weights
      # Note here that we are asserting to check variables must have 1 element.
      # If we use any variable in forward pass, it will be passed here to
      # compute gradient during backward pass.
      grad_vars = []
      assert variables is not None
      assert len(variables) == 1
      assert variables[0] is self.weights
      raw_dy_dw = [x ** (self.order-k) for k in range(self.order+1)]
      dy_dw = dpoly * tf.stack(raw_dy_dw)
      grad_vars.append(
        tf.reduce_sum(tf.reshape(dy_dw, [self.order+1, -1]), axis=1)
      )
      return grad_xs, grad_vars
    return poly, grad_fn
