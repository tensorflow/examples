# Copyright 2020 Google Research. All Rights Reserved.
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
"""Contains definitions for EfficientNet model.

[1] Mingxing Tan, Quoc V. Le
  EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.
  ICML'19, https://arxiv.org/abs/1905.11946
"""

import collections
import itertools
import math

from absl import logging
import numpy as np
import six
from six.moves import xrange
import tensorflow as tf

from tensorflow_examples.lite.model_maker.third_party.efficientdet import utils

GlobalParams = collections.namedtuple('GlobalParams', [
    'batch_norm_momentum', 'batch_norm_epsilon', 'dropout_rate', 'data_format',
    'num_classes', 'width_coefficient', 'depth_coefficient', 'depth_divisor',
    'min_depth', 'survival_prob', 'relu_fn', 'batch_norm', 'use_se',
    'local_pooling', 'condconv_num_experts', 'clip_projection_output',
    'blocks_args', 'fix_head_stem', 'grad_checkpoint'
])
GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)

BlockArgs = collections.namedtuple('BlockArgs', [
    'kernel_size', 'num_repeat', 'input_filters', 'output_filters',
    'expand_ratio', 'id_skip', 'strides', 'se_ratio', 'conv_type', 'fused_conv',
    'super_pixel', 'condconv'
])
# defaults will be a public argument for namedtuple in Python 3.7
# https://docs.python.org/3/library/collections.html#collections.namedtuple
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)


def conv_kernel_initializer(shape, dtype=None, partition_info=None):
  """Initialization for convolutional kernels.

  The main difference with tf.variance_scaling_initializer is that
  tf.variance_scaling_initializer uses a truncated normal with an uncorrected
  standard deviation, whereas here we use a normal distribution. Similarly,
  tf.initializers.variance_scaling uses a truncated normal with
  a corrected standard deviation.

  Args:
    shape: shape of variable
    dtype: dtype of variable
    partition_info: unused

  Returns:
    an initialization for the variable
  """
  del partition_info
  kernel_height, kernel_width, _, out_filters = shape
  fan_out = int(kernel_height * kernel_width * out_filters)
  return tf.random.normal(
      shape, mean=0.0, stddev=np.sqrt(2.0 / fan_out), dtype=dtype)


def dense_kernel_initializer(shape, dtype=None, partition_info=None):
  """Initialization for dense kernels.

  This initialization is equal to
    tf.variance_scaling_initializer(scale=1.0/3.0, mode='fan_out',
                                    distribution='uniform').
  It is written out explicitly here for clarity.

  Args:
    shape: shape of variable
    dtype: dtype of variable
    partition_info: unused

  Returns:
    an initialization for the variable
  """
  del partition_info
  init_range = 1.0 / np.sqrt(shape[1])
  return tf.random.uniform(shape, -init_range, init_range, dtype=dtype)


def superpixel_kernel_initializer(shape, dtype='float32', partition_info=None):
  """Initializes superpixel kernels.

  This is inspired by space-to-depth transformation that is mathematically
  equivalent before and after the transformation. But we do the space-to-depth
  via a convolution. Moreover, we make the layer trainable instead of direct
  transform, we can initialization it this way so that the model can learn not
  to do anything but keep it mathematically equivalent, when improving
  performance.


  Args:
    shape: shape of variable
    dtype: dtype of variable
    partition_info: unused

  Returns:
    an initialization for the variable
  """
  del partition_info
  #  use input depth to make superpixel kernel.
  depth = shape[-2]
  filters = np.zeros([2, 2, depth, 4 * depth], dtype=dtype)
  i = np.arange(2)
  j = np.arange(2)
  k = np.arange(depth)
  mesh = np.array(np.meshgrid(i, j, k)).T.reshape(-1, 3).T
  filters[mesh[0], mesh[1], mesh[2], 4 * mesh[2] + 2 * mesh[0] + mesh[1]] = 1
  return filters


def round_filters(filters, global_params, skip=False):
  """Round number of filters based on depth multiplier."""
  multiplier = global_params.width_coefficient
  divisor = global_params.depth_divisor
  min_depth = global_params.min_depth
  if skip or not multiplier:
    return filters

  filters *= multiplier
  min_depth = min_depth or divisor
  new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
  # Make sure that round down does not go down by more than 10%.
  if new_filters < 0.9 * filters:
    new_filters += divisor
  return int(new_filters)


def round_repeats(repeats, global_params, skip=False):
  """Round number of filters based on depth multiplier."""
  multiplier = global_params.depth_coefficient
  if skip or not multiplier:
    return repeats
  return int(math.ceil(multiplier * repeats))


class SE(tf.keras.layers.Layer):
  """Squeeze-and-excitation layer."""

  def __init__(self, global_params, se_filters, output_filters, name=None):
    super().__init__(name=name)

    self._local_pooling = global_params.local_pooling
    self._data_format = global_params.data_format
    self._relu_fn = global_params.relu_fn or tf.nn.swish

    # Squeeze and Excitation layer.
    self._se_reduce = tf.keras.layers.Conv2D(
        se_filters,
        kernel_size=[1, 1],
        strides=[1, 1],
        kernel_initializer=conv_kernel_initializer,
        padding='same',
        data_format=self._data_format,
        use_bias=True,
        name='conv2d')
    self._se_expand = tf.keras.layers.Conv2D(
        output_filters,
        kernel_size=[1, 1],
        strides=[1, 1],
        kernel_initializer=conv_kernel_initializer,
        padding='same',
        data_format=self._data_format,
        use_bias=True,
        name='conv2d_1')

  def call(self, inputs):
    h_axis, w_axis = [2, 3] if self._data_format == 'channels_first' else [1, 2]
    if self._local_pooling:
      se_tensor = tf.nn.avg_pool(
          inputs,
          ksize=[1, inputs.shape[h_axis], inputs.shape[w_axis], 1],
          strides=[1, 1, 1, 1],
          padding='VALID')
    else:
      se_tensor = tf.reduce_mean(inputs, [h_axis, w_axis], keepdims=True)
    se_tensor = self._se_expand(self._relu_fn(self._se_reduce(se_tensor)))
    logging.info('Built SE %s : %s', self.name, se_tensor.shape)
    return tf.sigmoid(se_tensor) * inputs


class SuperPixel(tf.keras.layers.Layer):
  """Super pixel layer."""

  def __init__(self, block_args, global_params, name=None):
    super().__init__(name=name)
    self._superpixel = tf.keras.layers.Conv2D(
        block_args.input_filters,
        kernel_size=[2, 2],
        strides=[2, 2],
        kernel_initializer=conv_kernel_initializer,
        padding='same',
        data_format=global_params.data_format,
        use_bias=False,
        name='conv2d')
    self._bnsp = global_params.batch_norm(
        axis=1 if global_params.data_format == 'channels_first' else -1,
        momentum=global_params.batch_norm_momentum,
        epsilon=global_params.batch_norm_epsilon,
        name='tpu_batch_normalization')
    self._relu_fn = global_params.relu_fn or tf.nn.swish

  def call(self, inputs, training):
    return self._relu_fn(self._bnsp(self._superpixel(inputs), training))


class MBConvBlock(tf.keras.layers.Layer):
  """A class of MBConv: Mobile Inverted Residual Bottleneck.

  Attributes:
    endpoints: dict. A list of internal tensors.
  """

  def __init__(self, block_args, global_params, name=None):
    """Initializes a MBConv block.

    Args:
      block_args: BlockArgs, arguments to create a Block.
      global_params: GlobalParams, a set of global parameters.
      name: layer name.
    """
    super().__init__(name=name)

    self._block_args = block_args
    self._global_params = global_params
    self._local_pooling = global_params.local_pooling
    self._batch_norm_momentum = global_params.batch_norm_momentum
    self._batch_norm_epsilon = global_params.batch_norm_epsilon
    self._batch_norm = global_params.batch_norm
    self._condconv_num_experts = global_params.condconv_num_experts
    self._data_format = global_params.data_format
    self._channel_axis = 1 if self._data_format == 'channels_first' else -1

    self._relu_fn = global_params.relu_fn or tf.nn.swish
    self._has_se = (
        global_params.use_se and self._block_args.se_ratio is not None and
        0 < self._block_args.se_ratio <= 1)

    self._clip_projection_output = global_params.clip_projection_output

    self.endpoints = None

    if self._block_args.condconv:
      raise ValueError('Condconv is not supported.')

    # Builds the block accordings to arguments.
    self._build()

  @property
  def block_args(self):
    return self._block_args

  def _build(self):
    """Builds block according to the arguments."""
    # pylint: disable=g-long-lambda
    bid = itertools.count(0)
    get_bn_name = lambda: 'tpu_batch_normalization' + ('' if not next(
        bid) else '_' + str(next(bid) // 2))
    cid = itertools.count(0)
    get_conv_name = lambda: 'conv2d' + ('' if not next(cid) else '_' + str(
        next(cid) // 2))
    # pylint: enable=g-long-lambda

    if self._block_args.super_pixel == 1:
      self.super_pixel = SuperPixel(
          self._block_args, self._global_params, name='super_pixel')
    else:
      self.super_pixel = None

    filters = self._block_args.input_filters * self._block_args.expand_ratio
    kernel_size = self._block_args.kernel_size

    if self._block_args.fused_conv:
      # Fused expansion phase. Called if using fused convolutions.
      self._fused_conv = tf.keras.layers.Conv2D(
          filters=filters,
          kernel_size=[kernel_size, kernel_size],
          strides=self._block_args.strides,
          kernel_initializer=conv_kernel_initializer,
          padding='same',
          data_format=self._data_format,
          use_bias=False,
          name=get_conv_name())
    else:
      # Expansion phase. Called if not using fused convolutions and expansion
      # phase is necessary.
      if self._block_args.expand_ratio != 1:
        self._expand_conv = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=[1, 1],
            strides=[1, 1],
            kernel_initializer=conv_kernel_initializer,
            padding='same',
            data_format=self._data_format,
            use_bias=False,
            name=get_conv_name())
        self._bn0 = self._batch_norm(
            axis=self._channel_axis,
            momentum=self._batch_norm_momentum,
            epsilon=self._batch_norm_epsilon,
            name=get_bn_name())

      # Depth-wise convolution phase. Called if not using fused convolutions.
      self._depthwise_conv = tf.keras.layers.DepthwiseConv2D(
          kernel_size=[kernel_size, kernel_size],
          strides=self._block_args.strides,
          depthwise_initializer=conv_kernel_initializer,
          padding='same',
          data_format=self._data_format,
          use_bias=False,
          name='depthwise_conv2d')

    self._bn1 = self._batch_norm(
        axis=self._channel_axis,
        momentum=self._batch_norm_momentum,
        epsilon=self._batch_norm_epsilon,
        name=get_bn_name())

    if self._has_se:
      num_reduced_filters = max(
          1, int(self._block_args.input_filters * self._block_args.se_ratio))
      self._se = SE(
          self._global_params, num_reduced_filters, filters, name='se')
    else:
      self._se = None

    # Output phase.
    filters = self._block_args.output_filters
    self._project_conv = tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=[1, 1],
        strides=[1, 1],
        kernel_initializer=conv_kernel_initializer,
        padding='same',
        data_format=self._data_format,
        use_bias=False,
        name=get_conv_name())
    self._bn2 = self._batch_norm(
        axis=self._channel_axis,
        momentum=self._batch_norm_momentum,
        epsilon=self._batch_norm_epsilon,
        name=get_bn_name())

  def call(self, inputs, training, survival_prob=None):
    """Implementation of call().

    Args:
      inputs: the inputs tensor.
      training: boolean, whether the model is constructed for training.
      survival_prob: float, between 0 to 1, drop connect rate.

    Returns:
      A output tensor.
    """
    @utils.recompute_grad(self._global_params.grad_checkpoint)
    def _call(inputs):
      logging.info('Block %s input shape: %s', self.name, inputs.shape)
      x = inputs

      # creates conv 2x2 kernel
      if self.super_pixel:
        x = self.super_pixel(x, training)
        logging.info('SuperPixel %s: %s', self.name, x.shape)

      if self._block_args.fused_conv:
        # If use fused mbconv, skip expansion and use regular conv.
        x = self._relu_fn(self._bn1(self._fused_conv(x), training=training))
        logging.info('Conv2D shape: %s', x.shape)
      else:
        # Otherwise, first apply expansion and then apply depthwise conv.
        if self._block_args.expand_ratio != 1:
          x = self._relu_fn(self._bn0(self._expand_conv(x), training=training))
          logging.info('Expand shape: %s', x.shape)

        x = self._relu_fn(self._bn1(self._depthwise_conv(x), training=training))
        logging.info('DWConv shape: %s', x.shape)

      if self._se:
        x = self._se(x)

      self.endpoints = {'expansion_output': x}

      x = self._bn2(self._project_conv(x), training=training)
      # Add identity so that quantization-aware training can insert quantization
      # ops correctly.
      x = tf.identity(x)
      if self._clip_projection_output:
        x = tf.clip_by_value(x, -6, 6)
      if self._block_args.id_skip:
        if all(
            s == 1 for s in self._block_args.strides
        ) and self._block_args.input_filters == self._block_args.output_filters:
          # Apply only if skip connection presents.
          if survival_prob:
            x = utils.drop_connect(x, training, survival_prob)
          x = tf.add(x, inputs)
      logging.info('Project shape: %s', x.shape)
      return x

    return _call(inputs)


class MBConvBlockWithoutDepthwise(MBConvBlock):
  """MBConv-like block without depthwise convolution and squeeze-and-excite."""

  def _build(self):
    """Builds block according to the arguments."""
    filters = self._block_args.input_filters * self._block_args.expand_ratio
    # pylint: disable=g-long-lambda
    cid = itertools.count(0)
    get_conv_name = lambda: 'conv2d' + ('' if not next(cid) else '_' + str(
        next(cid) // 2))
    # pylint: enable=g-long-lambda
    kernel_size = self._block_args.kernel_size
    if self._block_args.expand_ratio != 1:
      # Expansion phase:
      self._expand_conv = tf.keras.layers.Conv2D(
          filters,
          kernel_size=[kernel_size, kernel_size],
          strides=[1, 1],
          kernel_initializer=conv_kernel_initializer,
          padding='same',
          use_bias=False,
          name=get_conv_name())
      self._bn0 = self._batch_norm(
          axis=self._channel_axis,
          momentum=self._batch_norm_momentum,
          epsilon=self._batch_norm_epsilon)

    # Output phase:
    filters = self._block_args.output_filters
    self._project_conv = tf.keras.layers.Conv2D(
        filters,
        kernel_size=[1, 1],
        strides=self._block_args.strides,
        kernel_initializer=conv_kernel_initializer,
        padding='same',
        use_bias=False,
        name=get_conv_name())
    self._bn1 = self._batch_norm(
        axis=self._channel_axis,
        momentum=self._batch_norm_momentum,
        epsilon=self._batch_norm_epsilon)

  def call(self, inputs, training, survival_prob=None):
    """Implementation of call().

    Args:
      inputs: the inputs tensor.
      training: boolean, whether the model is constructed for training.
      survival_prob: float, between 0 to 1, drop connect rate.

    Returns:
      A output tensor.
    """

    @utils.recompute_grad(self._global_params.grad_checkpoint)
    def _call(inputs):
      logging.info('Block %s  input shape: %s', self.name, inputs.shape)
      if self._block_args.expand_ratio != 1:
        x = self._relu_fn(
            self._bn0(self._expand_conv(inputs), training=training))
      else:
        x = inputs
      logging.info('Expand shape: %s', x.shape)

      self.endpoints = {'expansion_output': x}

      x = self._bn1(self._project_conv(x), training=training)
      # Add identity so that quantization-aware training can insert quantization
      # ops correctly.
      x = tf.identity(x)
      if self._clip_projection_output:
        x = tf.clip_by_value(x, -6, 6)

      if self._block_args.id_skip:
        if all(
            s == 1 for s in self._block_args.strides
        ) and self._block_args.input_filters == self._block_args.output_filters:
          # Apply only if skip connection presents.
          if survival_prob:
            x = utils.drop_connect(x, training, survival_prob)
          x = tf.add(x, inputs)
      logging.info('Project shape: %s', x.shape)
      return x

    return _call(inputs)


class Stem(tf.keras.layers.Layer):
  """Stem layer at the begining of the network."""

  def __init__(self, global_params, stem_filters, name=None):
    super().__init__(name=name)
    self._conv_stem = tf.keras.layers.Conv2D(
        filters=round_filters(stem_filters, global_params,
                              global_params.fix_head_stem),
        kernel_size=[3, 3],
        strides=[2, 2],
        kernel_initializer=conv_kernel_initializer,
        padding='same',
        data_format=global_params.data_format,
        use_bias=False)
    self._bn = global_params.batch_norm(
        axis=(1 if global_params.data_format == 'channels_first' else -1),
        momentum=global_params.batch_norm_momentum,
        epsilon=global_params.batch_norm_epsilon)
    self._relu_fn = global_params.relu_fn or tf.nn.swish

  def call(self, inputs, training):
    return self._relu_fn(self._bn(self._conv_stem(inputs), training=training))


class Head(tf.keras.layers.Layer):
  """Head layer for network outputs."""

  def __init__(self, global_params, name=None):
    super().__init__(name=name)

    self.endpoints = {}
    self._global_params = global_params

    self._conv_head = tf.keras.layers.Conv2D(
        filters=round_filters(1280, global_params, global_params.fix_head_stem),
        kernel_size=[1, 1],
        strides=[1, 1],
        kernel_initializer=conv_kernel_initializer,
        padding='same',
        data_format=global_params.data_format,
        use_bias=False,
        name='conv2d')
    self._bn = global_params.batch_norm(
        axis=(1 if global_params.data_format == 'channels_first' else -1),
        momentum=global_params.batch_norm_momentum,
        epsilon=global_params.batch_norm_epsilon)
    self._relu_fn = global_params.relu_fn or tf.nn.swish

    self._avg_pooling = tf.keras.layers.GlobalAveragePooling2D(
        data_format=global_params.data_format)
    if global_params.num_classes:
      self._fc = tf.keras.layers.Dense(
          global_params.num_classes,
          kernel_initializer=dense_kernel_initializer)
    else:
      self._fc = None

    if global_params.dropout_rate > 0:
      self._dropout = tf.keras.layers.Dropout(global_params.dropout_rate)
    else:
      self._dropout = None

    self.h_axis, self.w_axis = (
        [2, 3] if global_params.data_format == 'channels_first' else [1, 2])

  def call(self, inputs, training, pooled_features_only):
    """Call the layer."""
    outputs = self._relu_fn(
        self._bn(self._conv_head(inputs), training=training))
    self.endpoints['head_1x1'] = outputs

    if self._global_params.local_pooling:
      shape = outputs.get_shape().as_list()
      kernel_size = [1, shape[self.h_axis], shape[self.w_axis], 1]
      outputs = tf.nn.avg_pool(
          outputs, ksize=kernel_size, strides=[1, 1, 1, 1], padding='VALID')
      self.endpoints['pooled_features'] = outputs
      if not pooled_features_only:
        if self._dropout:
          outputs = self._dropout(outputs, training=training)
        self.endpoints['global_pool'] = outputs
        if self._fc:
          outputs = tf.squeeze(outputs, [self.h_axis, self.w_axis])
          outputs = self._fc(outputs)
        self.endpoints['head'] = outputs
    else:
      outputs = self._avg_pooling(outputs)
      self.endpoints['pooled_features'] = outputs
      if not pooled_features_only:
        if self._dropout:
          outputs = self._dropout(outputs, training=training)
        self.endpoints['global_pool'] = outputs
        if self._fc:
          outputs = self._fc(outputs)
        self.endpoints['head'] = outputs
    return outputs


class Model(tf.keras.Model):
  """A class implements tf.keras.Model.

    Reference: https://arxiv.org/abs/1807.11626
  """

  def __init__(self, blocks_args=None, global_params=None, name=None):
    """Initializes an `Model` instance.

    Args:
      blocks_args: A list of BlockArgs to construct block modules.
      global_params: GlobalParams, a set of global parameters.
      name: A string of layer name.

    Raises:
      ValueError: when blocks_args is not specified as a list.
    """
    super().__init__(name=name)

    if not isinstance(blocks_args, list):
      raise ValueError('blocks_args should be a list.')
    self._global_params = global_params
    self._blocks_args = blocks_args
    self._relu_fn = global_params.relu_fn or tf.nn.swish
    self._batch_norm = global_params.batch_norm
    self._fix_head_stem = global_params.fix_head_stem

    self.endpoints = None

    self._build()

  def _get_conv_block(self, conv_type):
    conv_block_map = {0: MBConvBlock, 1: MBConvBlockWithoutDepthwise}
    return conv_block_map[conv_type]

  def _build(self):
    """Builds a model."""
    self._blocks = []

    # Stem part.
    self._stem = Stem(self._global_params, self._blocks_args[0].input_filters)

    # Builds blocks.
    block_id = itertools.count(0)
    block_name = lambda: 'blocks_%d' % next(block_id)
    for i, block_args in enumerate(self._blocks_args):
      assert block_args.num_repeat > 0
      assert block_args.super_pixel in [0, 1, 2]
      # Update block input and output filters based on depth multiplier.
      input_filters = round_filters(block_args.input_filters,
                                    self._global_params)

      output_filters = round_filters(block_args.output_filters,
                                     self._global_params)
      kernel_size = block_args.kernel_size
      if self._fix_head_stem and (i == 0 or i == len(self._blocks_args) - 1):
        repeats = block_args.num_repeat
      else:
        repeats = round_repeats(block_args.num_repeat, self._global_params)
      block_args = block_args._replace(
          input_filters=input_filters,
          output_filters=output_filters,
          num_repeat=repeats)

      # The first block needs to take care of stride and filter size increase.
      conv_block = self._get_conv_block(block_args.conv_type)
      if not block_args.super_pixel:  #  no super_pixel at all
        self._blocks.append(
            conv_block(block_args, self._global_params, name=block_name()))
      else:
        # if superpixel, adjust filters, kernels, and strides.
        depth_factor = int(4 / block_args.strides[0] / block_args.strides[1])
        block_args = block_args._replace(
            input_filters=block_args.input_filters * depth_factor,
            output_filters=block_args.output_filters * depth_factor,
            kernel_size=((block_args.kernel_size + 1) //
                         2 if depth_factor > 1 else block_args.kernel_size))
        # if the first block has stride-2 and super_pixel trandformation
        if (block_args.strides[0] == 2 and block_args.strides[1] == 2):
          block_args = block_args._replace(strides=[1, 1])
          self._blocks.append(
              conv_block(block_args, self._global_params, name=block_name()))
          block_args = block_args._replace(  # sp stops at stride-2
              super_pixel=0,
              input_filters=input_filters,
              output_filters=output_filters,
              kernel_size=kernel_size)
        elif block_args.super_pixel == 1:
          self._blocks.append(
              conv_block(block_args, self._global_params, name=block_name()))
          block_args = block_args._replace(super_pixel=2)
        else:
          self._blocks.append(
              conv_block(block_args, self._global_params, name=block_name()))
      if block_args.num_repeat > 1:  # rest of blocks with the same block_arg
        # pylint: disable=protected-access
        block_args = block_args._replace(
            input_filters=block_args.output_filters, strides=[1, 1])
        # pylint: enable=protected-access
      for _ in xrange(block_args.num_repeat - 1):
        self._blocks.append(
            conv_block(block_args, self._global_params, name=block_name()))

    # Head part.
    self._head = Head(self._global_params)

  def call(self,
           inputs,
           training,
           features_only=None,
           pooled_features_only=False):
    """Implementation of call().

    Args:
      inputs: input tensors.
      training: boolean, whether the model is constructed for training.
      features_only: build the base feature network only.
      pooled_features_only: build the base network for features extraction
        (after 1x1 conv layer and global pooling, but before dropout and fc
        head).

    Returns:
      output tensors.
    """
    outputs = None
    # Keep the original inputs in endpoints for U-Net style FPN.
    self.endpoints = {'inputs': inputs}
    reduction_idx = 0

    # Calls Stem layers
    outputs = self._stem(inputs, training)
    logging.info('Built stem %s : %s', self._stem.name, outputs.shape)
    self.endpoints['stem'] = outputs

    # Calls blocks.
    for idx, block in enumerate(self._blocks):
      is_reduction = False  # reduction flag for blocks after the stem layer
      # If the first block has super-pixel (space-to-depth) layer, then stem is
      # the first reduction point.
      if (block.block_args.super_pixel == 1 and idx == 0):
        reduction_idx += 1
        self.endpoints['reduction_%s' % reduction_idx] = outputs

      elif ((idx == len(self._blocks) - 1) or
            self._blocks[idx + 1].block_args.strides[0] > 1):
        is_reduction = True
        reduction_idx += 1

      survival_prob = self._global_params.survival_prob
      if survival_prob:
        drop_rate = 1.0 - survival_prob
        survival_prob = 1.0 - drop_rate * float(idx) / len(self._blocks)
        logging.info('block_%s survival_prob: %s', idx, survival_prob)
      outputs = block(outputs, training=training, survival_prob=survival_prob)
      self.endpoints['block_%s' % idx] = outputs
      if is_reduction:
        self.endpoints['reduction_%s' % reduction_idx] = outputs
      if block.endpoints:
        for k, v in six.iteritems(block.endpoints):
          self.endpoints['block_%s/%s' % (idx, k)] = v
          if is_reduction:
            self.endpoints['reduction_%s/%s' % (reduction_idx, k)] = v
    self.endpoints['features'] = outputs

    if not features_only:
      # Calls final layers and returns logits.
      outputs = self._head(outputs, training, pooled_features_only)
      self.endpoints.update(self._head.endpoints)

    return [outputs] + list(
        filter(lambda endpoint: endpoint is not None, [
            self.endpoints.get('reduction_1'),
            self.endpoints.get('reduction_2'),
            self.endpoints.get('reduction_3'),
            self.endpoints.get('reduction_4'),
            self.endpoints.get('reduction_5'),
        ]))
