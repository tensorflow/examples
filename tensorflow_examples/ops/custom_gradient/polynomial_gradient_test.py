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
"""PolynomialGradient Tests"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_examples.ops.custom_gradient import polynomial_gradient


class PolynomialGradientTest(tf.test.TestCase):

  def test_scalar_input(self):
    y, dy_dx, dy_dw = compute_gradient(
      tf.Variable(tf.ones([2])), 1,
      tf.constant(2.)
    )
    self.assertAllClose(y, 3.)
    self.assertAllClose(dy_dx, 1.0)
    self.assertAllClose(dy_dw, [2., 1.])

  def test_vector_input(self):
    y, dy_dx, dy_dw = compute_gradient(
      tf.Variable(tf.ones([2])), 1,
      tf.constant([1., 2.])
    )
    self.assertAllClose(y, [2., 3.])
    self.assertAllClose(dy_dx, [1., 1.])
    self.assertAllClose(dy_dw, [3., 2.])

  def test_order_one(self):
    y, dy_dx, dy_dw = compute_gradient(
      tf.Variable(tf.ones([2])), 1,
      tf.constant([1., 2., 3.])
    )
    self.assertAllClose(y, [2., 3., 4.])
    self.assertAllClose(dy_dx, [1., 1., 1.])
    self.assertAllClose(dy_dw, [6., 3.])

  def test_order_higher(self):
    y, dy_dx, dy_dw = compute_gradient(
      tf.Variable(tf.ones([5])), 4,
      tf.constant([5., 4., 3., 2., 1.])
    )
    self.assertAllClose(y, [781., 341., 121., 31., 5.])
    self.assertAllClose(dy_dx, [586., 313., 142., 49., 10.])
    self.assertAllClose(dy_dw, [979., 225., 55., 15., 5.])

  def test_weights(self):
    y, dy_dx, dy_dw = compute_gradient(
      tf.Variable([3.5, 2.25, 1.2, 6.4]), 3,
      tf.constant([2.4, 7.5, 9.6, 12.5])
    )
    self.assertAllClose(y, [70.62401, 1618.525, 3321.8562, 7208.9])
    self.assertAllClose(dy_dx, [72.48, 625.575, 1012.0801, 1698.075])
    self.assertAllClose(dy_dw, [3273.56, 310.42, 32., 4.])

def compute_gradient(weights, order, input):
  with tf.GradientTape(persistent=True) as tape:
    tape.watch(input)
    pg = polynomial_gradient.PolynomialGradient(weights, order)
    poly = pg.calc_gradient(input)
  dy_dx = tape.gradient(poly, input)
  dy_dw = tape.gradient(poly, weights)
  return poly, dy_dx, dy_dw

if __name__ == "__main__":
  tf.test.main()
