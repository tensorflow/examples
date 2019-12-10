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
"""Head model configuration for simple softmax classifiers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.compat import v1 as tfv1


class SoftmaxClassifierHead(object):
  """Head model configuration for a fixed classifier model architecture.

  This configuration does not require defining a custom model.
  It can be used when the head model should be a simple linear
  classifier: one fully-connected layer with softmax activation
  and cross-entropy loss function.

  This configuration can work without Flex runtime.
  """

  def __init__(self, train_batch_size, input_shape, num_classes, l2_reg=None):
    """Constructs a SoftmaxClassifierHead instance.

    Args:
      train_batch_size: batch size to be used during training.
      input_shape: shape of the bottleneck inputs to the model.
      num_classes: number of classes for the target classification task.
      l2_reg: lambda parameter for L2 weights regularization. Default is no
        regularization.
    """
    self._train_batch_size = train_batch_size
    self._input_shape = input_shape
    self._num_features = np.prod(input_shape)
    self._num_classes = num_classes
    self._l2_reg = l2_reg

  def predict(self, bottleneck, scope='head'):
    """Appends the serving signature of the model to the current graph.

    Bottleneck tensor is connected as an input to the added model.
    All model variables are converted to placeholders and returned
    in a list.

    Args:
      bottleneck: tensor in the current graph to be connected as input.
      scope: name of the scope to load the model into.

    Returns:
      (head model output tensor, list of variable placeholders)
    """
    logits, variables, _ = self._logits(bottleneck, scope)
    predictions = tf.nn.softmax(logits)
    return predictions, variables

  def train(self, bottleneck, labels, scope='head'):
    """Appends the train signature of the model to the current graph.

    Bottleneck and labels tensors are connected as inputs.
    All model variables are converted to placeholders and returned
    in a list.

    Args:
      bottleneck: tensor containing input bottlenecks.
      labels: tensor containing one-hot ground truth labels.
      scope: name of the scope to load the model into.

    Returns:
      (loss tensor, list of variable gradients, list of variable placeholders)
    """
    logits, [ws, bs], flat_bottleneck = self._logits(bottleneck, scope)
    with tf.name_scope(scope + '/loss'):
      predictions = tf.nn.softmax(logits)
      cross_entropy = -tf.reduce_sum(labels * tf.math.log(predictions), 1)
      loss = tf.reduce_mean(cross_entropy)
      if self._l2_reg is not None:
        loss += self._l2_reg * tf.reduce_sum(ws**2)
    with tf.name_scope(scope + '/backprop'):
      # d_bs is also equal to combined sigmoid and cross-entropy gradient.
      d_bs = predictions - labels
      flat_bottleneck_t = tf.transpose(flat_bottleneck)
      d_ws = tf.matmul(flat_bottleneck_t, d_bs) / self._train_batch_size
      d_bs = tf.reduce_mean(d_bs, 0)
      if self._l2_reg is not None:
        d_ws += 2 * self._l2_reg * ws
    return loss, [d_ws, d_bs], [ws, bs]

  def _logits(self, bottleneck, scope):
    """Appends the forward pass of the model."""
    with tfv1.variable_scope(scope):
      flat_bottleneck = tf.reshape(bottleneck, (-1, self._num_features))
      ws = tfv1.placeholder(
          tf.float32,
          shape=(self._num_features, self._num_classes),
          name='placeholder_ws')
      bs = tfv1.placeholder(
          tf.float32, shape=(self._num_classes,), name='placeholder_bs')
      logits = tf.matmul(flat_bottleneck, ws) + bs
      return logits, [ws, bs], flat_bottleneck

  def generate_initial_params(self):
    """Constructs a TF function that computes initial parameter values.

    The function accepts a single scalar input that should always be
    zero. Without this input, TFLiteConverter eagerly converts all
    tf.fill instances into constants, instead of emitting Fill ops.

    Returns:
      TensorFlow function that returns initial model parameter values.
    """

    @tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.float32)])
    def model_func(zero):
      ws = tf.fill((self._num_features, self._num_classes), zero)
      bs = tf.fill((self._num_classes,), zero)
      return ws, bs

    return model_func

  def input_shape(self):
    """Returns the model input shape."""
    return self._input_shape

  def train_requires_flex(self):
    """Whether the generated training model requires Flex support."""
    return False
