# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Densely Connected Convolutional Networks.

Reference [
Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

l2 = tf.keras.regularizers.l2


def calc_from_depth(depth, num_blocks, bottleneck):
  """Calculate number of layers in each block from the depth.

  Args:
    depth: Depth of model
    num_blocks: Number of dense blocks
    bottleneck: If True, num_layers will be halved

  Returns:
    Number of layers in each block as a list

  Raises:
    ValueError: If depth or num_blocks is None and num_blocks is not 3.
  """

  if depth is None or num_blocks is None:
    raise ValueError("For 'from_depth' mode, you need to specify the depth "
                     "and number of blocks.")

  if num_blocks != 3:
    raise ValueError(
        "Number of blocks must be 3 if mode is 'from_depth'.")

  if (depth - 4) % 3 == 0:
    num_layers = (depth - 4) / 3
    if bottleneck:
      num_layers //= 2
    return [num_layers] * num_blocks
  else:
    raise ValueError("Depth must be 3N+4 if mode is 'from_depth'.")


def calc_from_list(depth, num_blocks, layers_per_block):
  """Calculate number of layers in each block.

  Args:
    depth: Depth of model
    num_blocks: Number of dense blocks
    layers_per_block: Number of layers per block as a list or tuple

  Returns:
    Number of layers in each block as a list

  Raises:
    ValueError: If depth or num_blocks is not None and
                layers_per_block is None or not a list or tuple
  """
  if depth is not None or num_blocks is not None:
    raise ValueError("You don't have to specify the depth and number of "
                     "blocks when mode is 'from_list'")

  if layers_per_block is None or not isinstance(
      layers_per_block, list) or not isinstance(layers_per_block, tuple):
    raise ValueError("You must pass list or tuple when using 'from_list' mode.")

  if isinstance(layers_per_block, list) or isinstance(layers_per_block, tuple):
    return list(layers_per_block)


def calc_from_integer(depth, num_blocks, layers_per_block):
  """Calculate number of layers in each block.

  Args:
    depth: Depth of model
    num_blocks: Number of dense blocks
    layers_per_block: Number of layers per block as an integer.

  Returns:
    Number of layers in each block as a list

  Raises:
    ValueError: If depth is not None and
                num_blocks is None or layer_per_block is not an integer.
  """
  if depth is not None:
    raise ValueError("You don't have to specify the depth "
                     "when mode is 'from_integer'")

  if num_blocks is None or not isinstance(layers_per_block, int):
    raise ValueError("You must pass number of blocks or an integer to "
                     "layers in each block")

  return [layers_per_block] * num_blocks


class ConvBlock(tf.keras.Model):
  """Convolutional Block consisting of (batchnorm->relu->conv).

  Arguments:
    num_filters: number of filters passed to a convolutional layer.
    data_format: "channels_first" or "channels_last"
    bottleneck: if True, then a 1x1 Conv is performed followed by 3x3 Conv.
    weight_decay: weight decay
    dropout_rate: dropout rate.
  """

  def __init__(self, num_filters, data_format, bottleneck, weight_decay=1e-4,
               dropout_rate=0):
    super(ConvBlock, self).__init__()
    self.bottleneck = bottleneck

    axis = -1 if data_format == "channels_last" else 1
    inter_filter = num_filters * 4
    # don't forget to set use_bias=False when using batchnorm
    self.conv2 = tf.keras.layers.Conv2D(num_filters,
                                        (3, 3),
                                        padding="same",
                                        use_bias=False,
                                        data_format=data_format,
                                        kernel_initializer="he_normal",
                                        kernel_regularizer=l2(weight_decay))
    self.batchnorm1 = tf.keras.layers.BatchNormalization(axis=axis)
    self.dropout = tf.keras.layers.Dropout(dropout_rate)

    if self.bottleneck:
      self.conv1 = tf.keras.layers.Conv2D(inter_filter,
                                          (1, 1),
                                          padding="same",
                                          use_bias=False,
                                          data_format=data_format,
                                          kernel_initializer="he_normal",
                                          kernel_regularizer=l2(weight_decay))
      self.batchnorm2 = tf.keras.layers.BatchNormalization(axis=axis)

  def call(self, x, training=True):
    output = self.batchnorm1(x, training=training)

    if self.bottleneck:
      output = self.conv1(tf.nn.relu(output))
      output = self.batchnorm2(output, training=training)

    output = self.conv2(tf.nn.relu(output))
    output = self.dropout(output, training=training)

    return output


class TransitionBlock(tf.keras.Model):
  """Transition Block to reduce the number of features.

  Arguments:
    num_filters: number of filters passed to a convolutional layer.
    data_format: "channels_first" or "channels_last"
    weight_decay: weight decay
    dropout_rate: dropout rate.
  """

  def __init__(self, num_filters, data_format,
               weight_decay=1e-4, dropout_rate=0):
    super(TransitionBlock, self).__init__()
    axis = -1 if data_format == "channels_last" else 1

    self.batchnorm = tf.keras.layers.BatchNormalization(axis=axis)
    self.conv = tf.keras.layers.Conv2D(num_filters,
                                       (1, 1),
                                       padding="same",
                                       use_bias=False,
                                       data_format=data_format,
                                       kernel_initializer="he_normal",
                                       kernel_regularizer=l2(weight_decay))
    self.avg_pool = tf.keras.layers.AveragePooling2D(pool_size(2,2),data_format=data_format)

  def call(self, x, training=True):
    output = self.batchnorm(x, training=training)
    output = self.conv(tf.nn.relu(output))
    output = self.avg_pool(output)
    return output


class DenseBlock(tf.keras.Model):
  """Dense Block.

  It consists of ConvBlocks where each block's output is concatenated
  with its input.

  Arguments:
    num_layers: Number of layers in each block.
    growth_rate: number of filters to add per conv block.
    data_format: "channels_first" or "channels_last"
    bottleneck: boolean, that decides which part of ConvBlock to call.
    weight_decay: weight decay
    dropout_rate: dropout rate.
  """

  def __init__(self, num_layers, growth_rate, data_format, bottleneck,
               weight_decay=1e-4, dropout_rate=0):
    super(DenseBlock, self).__init__()
    self.num_layers = num_layers
    self.axis = -1 if data_format == "channels_last" else 1

    self.blocks = []
    for _ in range(int(self.num_layers)):
      self.blocks.append(ConvBlock(growth_rate,
                                   data_format,
                                   bottleneck,
                                   weight_decay,
                                   dropout_rate))

  def call(self, x, training=True):
    for i in range(int(self.num_layers)):
      output = self.blocks[i](x, training=training)
      x = tf.concat([x, output], axis=self.axis)

    return x


class DenseNet(tf.keras.Model):
  """Creating the Densenet Architecture.

  Arguments:
    mode: mode could be:
        - from_depth: num_layers_in_each_block will be calculated from the depth
                      and number of blocks.
        - from_list: pass num_layers_in_each_block as a list. depth and number
                     of blocks should not be specified
        - from_integer: pass num_layers_in_each_block as an integer. Number of
                        layers in each block will be calculated using
                        num_of_blocks * num_layers_in_each_block
    depth_of_model: number of layers in the model.
    growth_rate: number of filters to add per conv block.
    num_of_blocks: number of dense blocks.
    output_classes: number of output classes.
    num_layers_in_each_block: number of layers in each block.
                              If -1, then we calculate this by (depth-3)/4.
                              If positive integer, then it is used as the
                                number of layers per block.
                              If list or tuple, then this list is used directly.
    data_format: "channels_first" or "channels_last"
    bottleneck: boolean, to decide which part of conv block to call.
    compression: reducing the number of inputs(filters) to the transition block.
    weight_decay: weight decay
    rate: dropout rate.
    pool_initial: If True add a 7x7 conv with stride 2 followed by 3x3 maxpool
                  else, do a 3x3 conv with stride 1.
    include_top: If true, GlobalAveragePooling Layer and Dense layer are
                 included.
  """

  def __init__(self, mode, growth_rate, output_classes, depth_of_model=None,
               num_of_blocks=None, num_layers_in_each_block=None,
               data_format="channels_last", bottleneck=True, compression=0.5,
               weight_decay=1e-4, dropout_rate=0., pool_initial=False,
               include_top=True):
    super(DenseNet, self).__init__()
    self.mode = mode
    self.depth_of_model = depth_of_model
    self.growth_rate = growth_rate
    self.num_of_blocks = num_of_blocks
    self.output_classes = output_classes
    self.num_layers_in_each_block = num_layers_in_each_block
    self.data_format = data_format
    self.bottleneck = bottleneck
    self.compression = compression
    self.weight_decay = weight_decay
    self.dropout_rate = dropout_rate
    self.pool_initial = pool_initial
    self.include_top = include_top

    # deciding number of layers in each block
    if mode == "from_depth":
      self.num_layers_in_each_block = calc_from_depth(
          self.depth_of_model, self.num_of_blocks, self.bottleneck)
    elif mode == "from_list":
      self.num_layers_in_each_block = calc_from_list(
          self.depth_of_model, self.num_of_blocks,
          self.num_layers_in_each_block)
    elif mode == "from_integer":
      self.num_layers_in_each_block = calc_from_integer(
          self.depth_of_model, self.num_of_blocks,
          self.num_layers_in_each_block)

    axis = -1 if self.data_format == "channels_last" else 1

    # setting the filters and stride of the initial conv layer.
    if self.pool_initial:
      init_filters = (7, 7)
      stride = (2, 2)
    else:
      init_filters = (3, 3)
      stride = (1, 1)

    self.num_filters = 2 * self.growth_rate

    # first conv and pool layer
    self.conv1 = tf.keras.layers.Conv2D(self.num_filters,
                                        init_filters,
                                        strides=stride,
                                        padding="same",
                                        use_bias=False,
                                        data_format=self.data_format,
                                        kernel_initializer="he_normal",
                                        kernel_regularizer=l2(
                                            self.weight_decay))
    if self.pool_initial:
      self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3),
                                                strides=(2, 2),
                                                padding="same",
                                                data_format=self.data_format)
      self.batchnorm1 = tf.keras.layers.BatchNormalization(axis=axis)

    self.batchnorm2 = tf.keras.layers.BatchNormalization(axis=axis)

    # calculating the number of filters after each block
    num_filters_after_each_block = [self.num_filters]
    for i in range(1, self.num_of_blocks):
      temp_num_filters = num_filters_after_each_block[i-1] + (
          self.growth_rate * self.num_layers_in_each_block[i-1])
      # using compression to reduce the number of inputs to the
      # transition block
      temp_num_filters = int(temp_num_filters * compression)
      num_filters_after_each_block.append(temp_num_filters)

    # dense block initialization
    self.dense_blocks = []
    self.transition_blocks = []
    for i in range(self.num_of_blocks):
      self.dense_blocks.append(DenseBlock(self.num_layers_in_each_block[i],
                                          self.growth_rate,
                                          self.data_format,
                                          self.bottleneck,
                                          self.weight_decay,
                                          self.dropout_rate))
      if i+1 < self.num_of_blocks:
        self.transition_blocks.append(
            TransitionBlock(num_filters_after_each_block[i+1],
                            self.data_format,
                            self.weight_decay,
                            self.dropout_rate))

    # last pooling and fc layer
    if self.include_top:
      self.last_pool = tf.keras.layers.GlobalAveragePooling2D(
          data_format=self.data_format)
      self.classifier = tf.keras.layers.Dense(self.output_classes)

  def call(self, x, training=True):
    output = self.conv1(x)

    if self.pool_initial:
      output = self.batchnorm1(output, training=training)
      output = tf.nn.relu(output)
      output = self.pool1(output)

    for i in range(self.num_of_blocks - 1):
      output = self.dense_blocks[i](output, training=training)
      output = self.transition_blocks[i](output, training=training)

    output = self.dense_blocks[
        self.num_of_blocks - 1](output, training=training)
    output = self.batchnorm2(output, training=training)
    output = tf.nn.relu(output)

    if self.include_top:
      output = self.last_pool(output)
      output = self.classifier(output)

    return output
