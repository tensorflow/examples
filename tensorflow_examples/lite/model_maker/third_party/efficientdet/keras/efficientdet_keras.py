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
"""Keras implementation of efficientdet."""
import functools
from absl import logging
import numpy as np
import tensorflow as tf

from tensorflow_examples.lite.model_maker.third_party.efficientdet import dataloader
from tensorflow_examples.lite.model_maker.third_party.efficientdet import hparams_config
from tensorflow_examples.lite.model_maker.third_party.efficientdet import utils
from tensorflow_examples.lite.model_maker.third_party.efficientdet.backbone import backbone_factory
from tensorflow_examples.lite.model_maker.third_party.efficientdet.backbone import efficientnet_builder
from tensorflow_examples.lite.model_maker.third_party.efficientdet.keras import fpn_configs
from tensorflow_examples.lite.model_maker.third_party.efficientdet.keras import postprocess
from tensorflow_examples.lite.model_maker.third_party.efficientdet.keras import tfmot
from tensorflow_examples.lite.model_maker.third_party.efficientdet.keras import util_keras


def add_n(nodes):
  """A customized add_n to add up a list of tensors."""
  # tf.add_n is not supported by EdgeTPU, while tf.reduce_sum is not supported
  # by GPU and runs slow on EdgeTPU because of the 5-dimension op.
  with tf.name_scope('add_n'):
    new_node = nodes[0]
    for n in nodes[1:]:
      new_node = new_node + n
    return new_node


class FNode(tf.keras.layers.Layer):
  """A Keras Layer implementing BiFPN Node."""

  def __init__(self,
               feat_level,
               inputs_offsets,
               fpn_num_filters,
               apply_bn_for_resampling,
               is_training_bn,
               conv_after_downsample,
               conv_bn_act_pattern,
               separable_conv,
               act_type,
               strategy,
               weight_method,
               data_format,
               model_optimizations,
               name='fnode'):
    super().__init__(name=name)
    self.feat_level = feat_level
    self.inputs_offsets = inputs_offsets
    self.fpn_num_filters = fpn_num_filters
    self.apply_bn_for_resampling = apply_bn_for_resampling
    self.separable_conv = separable_conv
    self.act_type = act_type
    self.is_training_bn = is_training_bn
    self.conv_after_downsample = conv_after_downsample
    self.strategy = strategy
    self.data_format = data_format
    self.weight_method = weight_method
    self.conv_bn_act_pattern = conv_bn_act_pattern
    self.resample_layers = []
    self.vars = []
    self.model_optimizations = model_optimizations

  def fuse_features(self, nodes):
    """Fuse features from different resolutions and return a weighted sum.

    Args:
      nodes: a list of tensorflow features at different levels

    Returns:
      A tensor denoting the fused feature.
    """
    dtype = nodes[0].dtype

    if self.weight_method == 'attn':
      edge_weights = [tf.cast(var, dtype=dtype) for var in self.vars]
      normalized_weights = tf.nn.softmax(tf.stack(edge_weights))
      nodes = tf.stack(nodes, axis=-1)
      new_node = tf.reduce_sum(nodes * normalized_weights, -1)
    elif self.weight_method == 'fastattn':
      edge_weights = [
          tf.nn.relu(tf.cast(var, dtype=dtype)) for var in self.vars
      ]
      weights_sum = add_n(edge_weights)
      nodes = [
          nodes[i] * edge_weights[i] / (weights_sum + 0.0001)
          for i in range(len(nodes))
      ]
      new_node = add_n(nodes)
    elif self.weight_method == 'channel_attn':
      edge_weights = [tf.cast(var, dtype=dtype) for var in self.vars]
      normalized_weights = tf.nn.softmax(tf.stack(edge_weights, -1), axis=-1)
      nodes = tf.stack(nodes, axis=-1)
      new_node = tf.reduce_sum(nodes * normalized_weights, -1)
    elif self.weight_method == 'channel_fastattn':
      edge_weights = [
          tf.nn.relu(tf.cast(var, dtype=dtype)) for var in self.vars
      ]
      weights_sum = add_n(edge_weights)
      nodes = [
          nodes[i] * edge_weights[i] / (weights_sum + 0.0001)
          for i in range(len(nodes))
      ]
      new_node = add_n(nodes)
    elif self.weight_method == 'sum':
      new_node = add_n(nodes)
    else:
      raise ValueError('unknown weight_method %s' % self.weight_method)

    return new_node

  def _add_wsm(self, initializer, shape=None):
    for i, _ in enumerate(self.inputs_offsets):
      name = 'WSM' + ('' if i == 0 else '_' + str(i))
      self.vars.append(
          self.add_weight(initializer=initializer, name=name, shape=shape))

  def build(self, feats_shape):
    for i, input_offset in enumerate(self.inputs_offsets):
      name = 'resample_{}_{}_{}'.format(i, input_offset, len(feats_shape))
      self.resample_layers.append(
          ResampleFeatureMap(
              self.feat_level,
              self.fpn_num_filters,
              self.apply_bn_for_resampling,
              self.is_training_bn,
              self.conv_after_downsample,
              strategy=self.strategy,
              data_format=self.data_format,
              model_optimizations=self.model_optimizations,
              name=name))
    if self.weight_method == 'attn':
      self._add_wsm('ones')
    elif self.weight_method == 'fastattn':
      self._add_wsm('ones')
    elif self.weight_method == 'channel_attn':
      num_filters = int(self.fpn_num_filters)
      self._add_wsm(tf.ones, num_filters)
    elif self.weight_method == 'channel_fastattn':
      num_filters = int(self.fpn_num_filters)
      self._add_wsm(tf.ones, num_filters)
    self.op_after_combine = OpAfterCombine(
        self.is_training_bn,
        self.conv_bn_act_pattern,
        self.separable_conv,
        self.fpn_num_filters,
        self.act_type,
        self.data_format,
        self.strategy,
        self.model_optimizations,
        name='op_after_combine{}'.format(len(feats_shape)))
    self.built = True
    super().build(feats_shape)

  def call(self, feats, training):
    nodes = []
    for i, input_offset in enumerate(self.inputs_offsets):
      input_node = feats[input_offset]
      input_node = self.resample_layers[i](input_node, training, feats)
      nodes.append(input_node)
    new_node = self.fuse_features(nodes)
    new_node = self.op_after_combine(new_node)
    return feats + [new_node]


class OpAfterCombine(tf.keras.layers.Layer):
  """Operation after combining input features during feature fusiong."""

  def __init__(self,
               is_training_bn,
               conv_bn_act_pattern,
               separable_conv,
               fpn_num_filters,
               act_type,
               data_format,
               strategy,
               model_optimizations,
               name='op_after_combine'):
    super().__init__(name=name)
    self.conv_bn_act_pattern = conv_bn_act_pattern
    self.separable_conv = separable_conv
    self.fpn_num_filters = fpn_num_filters
    self.act_type = act_type
    self.data_format = data_format
    self.strategy = strategy
    self.is_training_bn = is_training_bn
    if self.separable_conv:
      conv2d_layer = functools.partial(
          tf.keras.layers.SeparableConv2D, depth_multiplier=1)
    else:
      conv2d_layer = tf.keras.layers.Conv2D

    self.conv_op = conv2d_layer(
        filters=fpn_num_filters,
        kernel_size=(3, 3),
        padding='same',
        use_bias=not self.conv_bn_act_pattern,
        data_format=self.data_format,
        name='conv')
    if model_optimizations:
      for method in model_optimizations.keys():
        self.conv_op = (
            tfmot.get_method(method)(self.conv_op))
    self.bn = util_keras.build_batch_norm(
        is_training_bn=self.is_training_bn,
        data_format=self.data_format,
        strategy=self.strategy,
        name='bn')

  def call(self, new_node, training):
    if not self.conv_bn_act_pattern:
      new_node = utils.activation_fn(new_node, self.act_type)
    new_node = self.conv_op(new_node)
    new_node = self.bn(new_node, training=training)
    if self.conv_bn_act_pattern:
      new_node = utils.activation_fn(new_node, self.act_type)
    return new_node


class ResampleFeatureMap(tf.keras.layers.Layer):
  """Resample feature map for downsampling or upsampling."""

  def __init__(self,
               feat_level,
               target_num_channels,
               apply_bn=False,
               is_training_bn=None,
               conv_after_downsample=False,
               strategy=None,
               data_format=None,
               pooling_type=None,
               upsampling_type=None,
               model_optimizations=None,
               name='resample_p0'):
    super().__init__(name=name)
    self.apply_bn = apply_bn
    self.is_training_bn = is_training_bn
    self.data_format = data_format
    self.target_num_channels = target_num_channels
    self.feat_level = feat_level
    self.strategy = strategy
    self.conv_after_downsample = conv_after_downsample
    self.pooling_type = pooling_type or 'max'
    self.upsampling_type = upsampling_type or 'nearest'

    self.conv2d = tf.keras.layers.Conv2D(
        self.target_num_channels, (1, 1),
        padding='same',
        data_format=self.data_format,
        name='conv2d')
    if model_optimizations:
      for method in model_optimizations.keys():
        self.conv2d = tfmot.get_method(method)(self.conv2d)
    self.bn = util_keras.build_batch_norm(
        is_training_bn=self.is_training_bn,
        data_format=self.data_format,
        strategy=self.strategy,
        name='bn')

  def _pool2d(self, inputs, height, width, target_height, target_width):
    """Pool the inputs to target height and width."""
    height_stride_size = int((height - 1) // target_height + 1)
    width_stride_size = int((width - 1) // target_width + 1)
    if self.pooling_type == 'max':
      return tf.keras.layers.MaxPooling2D(
          pool_size=[height_stride_size + 1, width_stride_size + 1],
          strides=[height_stride_size, width_stride_size],
          padding='SAME',
          data_format=self.data_format)(inputs)
    if self.pooling_type == 'avg':
      return tf.keras.layers.AveragePooling2D(
          pool_size=[height_stride_size + 1, width_stride_size + 1],
          strides=[height_stride_size, width_stride_size],
          padding='SAME',
          data_format=self.data_format)(inputs)
    raise ValueError('Unsupported pooling type {}.'.format(self.pooling_type))

  def _upsample2d(self, inputs, target_height, target_width):
    return tf.cast(
        tf.compat.v1.image.resize_nearest_neighbor(
            tf.cast(inputs, tf.float32), [target_height, target_width]),
        inputs.dtype)

  def _maybe_apply_1x1(self, feat, training, num_channels):
    """Apply 1x1 conv to change layer width if necessary."""
    if num_channels != self.target_num_channels:
      feat = self.conv2d(feat)
      if self.apply_bn:
        feat = self.bn(feat, training=training)
    return feat

  def call(self, feat, training, all_feats):
    hwc_idx = (2, 3, 1) if self.data_format == 'channels_first' else (1, 2, 3)
    height, width, num_channels = [feat.shape.as_list()[i] for i in hwc_idx]
    if all_feats:
      target_feat_shape = all_feats[self.feat_level].shape.as_list()
      target_height, target_width, _ = [target_feat_shape[i] for i in hwc_idx]
    else:
      # Default to downsampling if all_feats is empty.
      target_height, target_width = (height + 1) // 2, (width + 1) // 2

    # If conv_after_downsample is True, when downsampling, apply 1x1 after
    # downsampling for efficiency.
    if height > target_height and width > target_width:
      if not self.conv_after_downsample:
        feat = self._maybe_apply_1x1(feat, training, num_channels)
      feat = self._pool2d(feat, height, width, target_height, target_width)
      if self.conv_after_downsample:
        feat = self._maybe_apply_1x1(feat, training, num_channels)
    elif height <= target_height and width <= target_width:
      feat = self._maybe_apply_1x1(feat, training, num_channels)
      if height < target_height or width < target_width:
        feat = self._upsample2d(feat, target_height, target_width)
    else:
      raise ValueError(
          'Incompatible Resampling : feat shape {}x{} target_shape: {}x{}'
          .format(height, width, target_height, target_width))

    return feat


class ClassNet(tf.keras.layers.Layer):
  """Object class prediction network."""

  def __init__(self,
               num_classes=90,
               num_anchors=9,
               num_filters=32,
               min_level=3,
               max_level=7,
               is_training_bn=False,
               act_type='swish',
               repeats=4,
               separable_conv=True,
               survival_prob=None,
               strategy=None,
               data_format='channels_last',
               grad_checkpoint=False,
               name='class_net',
               feature_only=False,
               **kwargs):
    """Initialize the ClassNet.

    Args:
      num_classes: number of classes.
      num_anchors: number of anchors.
      num_filters: number of filters for "intermediate" layers.
      min_level: minimum level for features.
      max_level: maximum level for features.
      is_training_bn: True if we train the BatchNorm.
      act_type: String of the activation used.
      repeats: number of intermediate layers.
      separable_conv: True to use separable_conv instead of conv2D.
      survival_prob: if a value is set then drop connect will be used.
      strategy: string to specify training strategy for TPU/GPU/CPU.
      data_format: string of 'channel_first' or 'channels_last'.
      grad_checkpoint: bool, If true, apply grad checkpoint for saving memory.
      name: the name of this layerl.
      feature_only: build the base feature network only (excluding final class
        head).
      **kwargs: other parameters.
    """

    super().__init__(name=name, **kwargs)
    self.num_classes = num_classes
    self.num_anchors = num_anchors
    self.num_filters = num_filters
    self.min_level = min_level
    self.max_level = max_level
    self.repeats = repeats
    self.separable_conv = separable_conv
    self.is_training_bn = is_training_bn
    self.survival_prob = survival_prob
    self.act_type = act_type
    self.strategy = strategy
    self.data_format = data_format
    self.conv_ops = []
    self.bns = []
    self.grad_checkpoint = grad_checkpoint
    self.feature_only = feature_only

    conv2d_layer = self.conv2d_layer(separable_conv, data_format)
    for i in range(self.repeats):
      # If using SeparableConv2D
      self.conv_ops.append(
          conv2d_layer(
              self.num_filters,
              kernel_size=3,
              bias_initializer=tf.zeros_initializer(),
              activation=None,
              padding='same',
              name='class-%d' % i))

      bn_per_level = []
      for level in range(self.min_level, self.max_level + 1):
        bn_per_level.append(
            util_keras.build_batch_norm(
                is_training_bn=self.is_training_bn,
                strategy=self.strategy,
                data_format=self.data_format,
                name='class-%d-bn-%d' % (i, level),
            ))
      self.bns.append(bn_per_level)

    self.classes = self.classes_layer(
        conv2d_layer, num_classes, num_anchors, name='class-predict')

  @tf.autograph.experimental.do_not_convert
  def _conv_bn_act(self, image, i, level_id, training):
    conv_op = self.conv_ops[i]
    bn = self.bns[i][level_id]
    act_type = self.act_type

    @utils.recompute_grad(self.grad_checkpoint)
    def _call(image):
      original_image = image
      image = conv_op(image)
      image = bn(image, training=training)
      if self.act_type:
        image = utils.activation_fn(image, act_type)
      if i > 0 and self.survival_prob:
        image = utils.drop_connect(image, training, self.survival_prob)
        image = image + original_image
      return image

    return _call(image)

  def call(self, inputs, training, **kwargs):
    """Call ClassNet."""
    class_outputs = []
    for level_id in range(0, self.max_level - self.min_level + 1):
      image = inputs[level_id]
      for i in range(self.repeats):
        image = self._conv_bn_act(image, i, level_id, training)
      if self.feature_only:
        class_outputs.append(image)
      else:
        class_outputs.append(self.classes(image))

    return class_outputs

  @classmethod
  def conv2d_layer(cls, separable_conv, data_format):
    """Gets the conv2d layer in ClassNet class."""
    if separable_conv:
      conv2d_layer = functools.partial(
          tf.keras.layers.SeparableConv2D,
          depth_multiplier=1,
          data_format=data_format,
          pointwise_initializer=tf.initializers.variance_scaling(),
          depthwise_initializer=tf.initializers.variance_scaling())
    else:
      conv2d_layer = functools.partial(
          tf.keras.layers.Conv2D,
          data_format=data_format,
          kernel_initializer=tf.random_normal_initializer(stddev=0.01))
    return conv2d_layer

  @classmethod
  def classes_layer(cls, conv2d_layer, num_classes, num_anchors, name):
    """Gets the classes layer in ClassNet class."""
    return conv2d_layer(
        num_classes * num_anchors,
        kernel_size=3,
        bias_initializer=tf.constant_initializer(-np.log((1 - 0.01) / 0.01)),
        padding='same',
        name=name)


class BoxNet(tf.keras.layers.Layer):
  """Box regression network."""

  def __init__(self,
               num_anchors=9,
               num_filters=32,
               min_level=3,
               max_level=7,
               is_training_bn=False,
               act_type='swish',
               repeats=4,
               separable_conv=True,
               survival_prob=None,
               strategy=None,
               data_format='channels_last',
               grad_checkpoint=False,
               name='box_net',
               feature_only=False,
               **kwargs):
    """Initialize BoxNet.

    Args:
      num_anchors: number of  anchors used.
      num_filters: number of filters for "intermediate" layers.
      min_level: minimum level for features.
      max_level: maximum level for features.
      is_training_bn: True if we train the BatchNorm.
      act_type: String of the activation used.
      repeats: number of "intermediate" layers.
      separable_conv: True to use separable_conv instead of conv2D.
      survival_prob: if a value is set then drop connect will be used.
      strategy: string to specify training strategy for TPU/GPU/CPU.
      data_format: string of 'channel_first' or 'channels_last'.
      grad_checkpoint: bool, If true, apply grad checkpoint for saving memory.
      name: Name of the layer.
      feature_only: build the base feature network only (excluding box class
        head).
      **kwargs: other parameters.
    """

    super().__init__(name=name, **kwargs)

    self.num_anchors = num_anchors
    self.num_filters = num_filters
    self.min_level = min_level
    self.max_level = max_level
    self.repeats = repeats
    self.separable_conv = separable_conv
    self.is_training_bn = is_training_bn
    self.survival_prob = survival_prob
    self.act_type = act_type
    self.strategy = strategy
    self.data_format = data_format
    self.grad_checkpoint = grad_checkpoint
    self.feature_only = feature_only

    self.conv_ops = []
    self.bns = []

    for i in range(self.repeats):
      # If using SeparableConv2D
      if self.separable_conv:
        self.conv_ops.append(
            tf.keras.layers.SeparableConv2D(
                filters=self.num_filters,
                depth_multiplier=1,
                pointwise_initializer=tf.initializers.variance_scaling(),
                depthwise_initializer=tf.initializers.variance_scaling(),
                data_format=self.data_format,
                kernel_size=3,
                activation=None,
                bias_initializer=tf.zeros_initializer(),
                padding='same',
                name='box-%d' % i))
      # If using Conv2d
      else:
        self.conv_ops.append(
            tf.keras.layers.Conv2D(
                filters=self.num_filters,
                kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                data_format=self.data_format,
                kernel_size=3,
                activation=None,
                bias_initializer=tf.zeros_initializer(),
                padding='same',
                name='box-%d' % i))

      bn_per_level = []
      for level in range(self.min_level, self.max_level + 1):
        bn_per_level.append(
            util_keras.build_batch_norm(
                is_training_bn=self.is_training_bn,
                strategy=self.strategy,
                data_format=self.data_format,
                name='box-%d-bn-%d' % (i, level)))
      self.bns.append(bn_per_level)

      self.boxes = self.boxes_layer(
          separable_conv, num_anchors, data_format, name='box-predict')

  @tf.autograph.experimental.do_not_convert
  def _conv_bn_act(self, image, i, level_id, training):
    conv_op = self.conv_ops[i]
    bn = self.bns[i][level_id]
    act_type = self.act_type

    @utils.recompute_grad(self.grad_checkpoint)
    def _call(image):
      original_image = image
      image = conv_op(image)
      image = bn(image, training=training)
      if self.act_type:
        image = utils.activation_fn(image, act_type)
      if i > 0 and self.survival_prob:
        image = utils.drop_connect(image, training, self.survival_prob)
        image = image + original_image
      return image

    return _call(image)

  def call(self, inputs, training):
    """Call boxnet."""
    box_outputs = []
    for level_id in range(0, self.max_level - self.min_level + 1):
      image = inputs[level_id]
      for i in range(self.repeats):
        image = self._conv_bn_act(image, i, level_id, training)

      if self.feature_only:
        box_outputs.append(image)
      else:
        box_outputs.append(self.boxes(image))

    return box_outputs

  @classmethod
  def boxes_layer(cls, separable_conv, num_anchors, data_format, name):
    """Gets the conv2d layer in BoxNet class."""
    if separable_conv:
      return tf.keras.layers.SeparableConv2D(
          filters=4 * num_anchors,
          depth_multiplier=1,
          pointwise_initializer=tf.initializers.variance_scaling(),
          depthwise_initializer=tf.initializers.variance_scaling(),
          data_format=data_format,
          kernel_size=3,
          activation=None,
          bias_initializer=tf.zeros_initializer(),
          padding='same',
          name=name)
    else:
      return tf.keras.layers.Conv2D(
          filters=4 * num_anchors,
          kernel_initializer=tf.random_normal_initializer(stddev=0.01),
          data_format=data_format,
          kernel_size=3,
          activation=None,
          bias_initializer=tf.zeros_initializer(),
          padding='same',
          name=name)


class SegmentationHead(tf.keras.layers.Layer):
  """Keras layer for semantic segmentation head."""

  def __init__(self,
               num_classes,
               num_filters,
               min_level,
               max_level,
               data_format,
               is_training_bn,
               act_type,
               strategy,
               name='segmentation_head',
               **kwargs):
    """Initialize SegmentationHead.

    Args:
      num_classes: number of classes.
      num_filters: number of filters for "intermediate" layers.
      min_level: minimum level for features.
      max_level: maximum level for features.
      data_format: string of 'channel_first' or 'channels_last'.
      is_training_bn: True if we train the BatchNorm.
      act_type: String of the activation used.
      strategy: string to specify training strategy for TPU/GPU/CPU.
      name: string of name.
      **kwargs: other parameters.
    """
    super().__init__(name=name, **kwargs)
    self.act_type = act_type
    self.con2d_ts = []
    self.con2d_t_bns = []
    for level in range(max_level - min_level):
      self.con2d_ts.append(
          tf.keras.layers.Conv2DTranspose(
              num_filters,
              3,
              strides=2,
              padding='same',
              data_format=data_format,
              use_bias=False))
      self.con2d_t_bns.append(
          util_keras.build_batch_norm(
              is_training_bn=is_training_bn,
              data_format=data_format,
              strategy=strategy,
              name='bn_' + str(level)))
    self.head_transpose = tf.keras.layers.Conv2DTranspose(
        num_classes, 3, strides=2, padding='same')

  def call(self, feats, training):
    x = feats[-1]
    skips = list(reversed(feats[:-1]))

    for con2d_t, con2d_t_bn, skip in zip(self.con2d_ts, self.con2d_t_bns,
                                         skips):
      x = con2d_t(x)
      x = con2d_t_bn(x, training)
      x = utils.activation_fn(x, self.act_type)
      x = tf.concat([x, skip], axis=-1)

    # This is the last layer of the model
    return self.head_transpose(x)  # 64x64 -> 128x128


class FPNCells(tf.keras.layers.Layer):
  """FPN cells."""

  def __init__(self, config, name='fpn_cells'):
    super().__init__(name=name)
    self.config = config

    if config.fpn_config:
      self.fpn_config = config.fpn_config
    else:
      self.fpn_config = fpn_configs.get_fpn_config(config.fpn_name,
                                                   config.min_level,
                                                   config.max_level,
                                                   config.fpn_weight_method)

    self.cells = [
        FPNCell(self.config, name='cell_%d' % rep)
        for rep in range(self.config.fpn_cell_repeats)
    ]

  def call(self, feats, training):
    for cell in self.cells:
      cell_feats = cell(feats, training)
      min_level = self.config.min_level
      max_level = self.config.max_level

      feats = []
      for level in range(min_level, max_level + 1):
        for i, fnode in enumerate(reversed(self.fpn_config.nodes)):
          if fnode['feat_level'] == level:
            feats.append(cell_feats[-1 - i])
            break

    return feats


class FPNCell(tf.keras.layers.Layer):
  """A single FPN cell."""

  def __init__(self, config, name='fpn_cell'):
    super().__init__(name=name)
    self.config = config
    if config.fpn_config:
      self.fpn_config = config.fpn_config
    else:
      self.fpn_config = fpn_configs.get_fpn_config(config.fpn_name,
                                                   config.min_level,
                                                   config.max_level,
                                                   config.fpn_weight_method)
    self.fnodes = []
    for i, fnode_cfg in enumerate(self.fpn_config.nodes):
      logging.info('fnode %d : %s', i, fnode_cfg)
      fnode = FNode(
          fnode_cfg['feat_level'] - self.config.min_level,
          fnode_cfg['inputs_offsets'],
          config.fpn_num_filters,
          config.apply_bn_for_resampling,
          config.is_training_bn,
          config.conv_after_downsample,
          config.conv_bn_act_pattern,
          config.separable_conv,
          config.act_type,
          strategy=config.strategy,
          weight_method=self.fpn_config.weight_method,
          data_format=config.data_format,
          model_optimizations=config.model_optimizations,
          name='fnode%d' % i)
      self.fnodes.append(fnode)

  def call(self, feats, training):
    @utils.recompute_grad(self.config.grad_checkpoint)
    def _call(feats):
      for fnode in self.fnodes:
        feats = fnode(feats, training)
      return feats
    return _call(feats)


class EfficientDetNet(tf.keras.Model):
  """EfficientDet keras network without pre/post-processing."""

  def __init__(self,
               model_name=None,
               config=None,
               name='',
               feature_only=False):
    """Initialize model."""
    super().__init__(name=name)

    config = config or hparams_config.get_efficientdet_config(model_name)
    self.config = config

    # Backbone.
    backbone_name = config.backbone_name
    is_training_bn = config.is_training_bn
    if 'efficientnet' in backbone_name:
      override_params = {
          'batch_norm':
              utils.batch_norm_class(is_training_bn, config.strategy),
          'relu_fn':
              functools.partial(utils.activation_fn, act_type=config.act_type),
          'grad_checkpoint': self.config.grad_checkpoint
      }
      if 'b0' in backbone_name:
        override_params['survival_prob'] = 0.0
      if config.backbone_config is not None:
        override_params['blocks_args'] = (
            efficientnet_builder.BlockDecoder().encode(
                config.backbone_config.blocks))
      override_params['data_format'] = config.data_format
      self.backbone = backbone_factory.get_model(
          backbone_name, override_params=override_params)

    # Feature network.
    self.resample_layers = []  # additional resampling layers.
    for level in range(6, config.max_level + 1):
      # Adds a coarser level by downsampling the last feature map.
      self.resample_layers.append(
          ResampleFeatureMap(
              feat_level=(level - config.min_level),
              target_num_channels=config.fpn_num_filters,
              apply_bn=config.apply_bn_for_resampling,
              is_training_bn=config.is_training_bn,
              conv_after_downsample=config.conv_after_downsample,
              strategy=config.strategy,
              data_format=config.data_format,
              model_optimizations=config.model_optimizations,
              name='resample_p%d' % level,
          ))
    self.fpn_cells = FPNCells(config)

    # class/box output prediction network.
    num_anchors = len(config.aspect_ratios) * config.num_scales
    num_filters = config.fpn_num_filters
    for head in config.heads:
      if head == 'object_detection':
        self.class_net = ClassNet(
            num_classes=config.num_classes,
            num_anchors=num_anchors,
            num_filters=num_filters,
            min_level=config.min_level,
            max_level=config.max_level,
            is_training_bn=config.is_training_bn,
            act_type=config.act_type,
            repeats=config.box_class_repeats,
            separable_conv=config.separable_conv,
            survival_prob=config.survival_prob,
            strategy=config.strategy,
            grad_checkpoint=config.grad_checkpoint,
            data_format=config.data_format,
            feature_only=feature_only)

        self.box_net = BoxNet(
            num_anchors=num_anchors,
            num_filters=num_filters,
            min_level=config.min_level,
            max_level=config.max_level,
            is_training_bn=config.is_training_bn,
            act_type=config.act_type,
            repeats=config.box_class_repeats,
            separable_conv=config.separable_conv,
            survival_prob=config.survival_prob,
            strategy=config.strategy,
            grad_checkpoint=config.grad_checkpoint,
            data_format=config.data_format,
            feature_only=feature_only)

      if head == 'segmentation':
        self.seg_head = SegmentationHead(
            num_classes=config.seg_num_classes,
            num_filters=num_filters,
            min_level=config.min_level,
            max_level=config.max_level,
            is_training_bn=config.is_training_bn,
            act_type=config.act_type,
            strategy=config.strategy,
            data_format=config.data_format)

  def _init_set_name(self, name, zero_based=True):
    """A hack to allow empty model name for legacy checkpoint compitability."""
    if name == '':  # pylint: disable=g-explicit-bool-comparison
      self._name = name
    else:
      self._name = super().__init__(name, zero_based)

  def call(self, inputs, training):
    config = self.config
    # call backbone network.
    all_feats = self.backbone(inputs, training=training, features_only=True)
    feats = all_feats[config.min_level:config.max_level + 1]

    # Build additional input features that are not from backbone.
    for resample_layer in self.resample_layers:
      feats.append(resample_layer(feats[-1], training, None))

    # call feature network.
    fpn_feats = self.fpn_cells(feats, training)

    # call class/box/seg output network.
    outputs = []
    if 'object_detection' in config.heads:
      class_outputs = self.class_net(fpn_feats, training)
      box_outputs = self.box_net(fpn_feats, training)
      outputs.extend([class_outputs, box_outputs])
    if 'segmentation' in config.heads:
      seg_outputs = self.seg_head(fpn_feats, training)
      outputs.append(seg_outputs)
    return tuple(outputs)


class EfficientDetModel(EfficientDetNet):
  """EfficientDet full keras model with pre and post processing."""

  def _preprocessing(self,
                     raw_images,
                     image_size,
                     mean_rgb,
                     stddev_rgb,
                     mode=None):
    """Preprocess images before feeding to the network."""
    if not mode:
      return raw_images, None

    image_size = utils.parse_image_size(image_size)
    if mode != 'infer':
      # We only support inference for now.
      raise ValueError('preprocessing must be infer or empty')

    def map_fn(image):
      input_processor = dataloader.DetectionInputProcessor(
          image, image_size)
      input_processor.normalize_image(mean_rgb, stddev_rgb)
      input_processor.set_scale_factors_to_output_size()
      image = input_processor.resize_and_crop_image()
      image_scale = input_processor.image_scale_to_original
      return image, image_scale

    if raw_images.shape.as_list()[0]:  # fixed batch size.
      batch_size = raw_images.shape.as_list()[0]
      outputs = [map_fn(raw_images[i]) for i in range(batch_size)]
      return [tf.stack(y) for y in zip(*outputs)]

    # otherwise treat it as dynamic batch size.
    return tf.vectorized_map(map_fn, raw_images)

  def _postprocess(self, cls_outputs, box_outputs, scales, mode='global'):
    """Postprocess class and box predictions."""
    if not mode:
      return cls_outputs, box_outputs

    if mode == 'global':
      return postprocess.postprocess_global(self.config.as_dict(), cls_outputs,
                                            box_outputs, scales)
    if mode == 'per_class':
      return postprocess.postprocess_per_class(self.config.as_dict(),
                                               cls_outputs, box_outputs, scales)
    if mode == 'combined':
      return postprocess.postprocess_combined(self.config.as_dict(),
                                              cls_outputs, box_outputs, scales)
    if mode == 'tflite':
      if scales is not None:
        # pre_mode should be None for TFLite.
        raise ValueError('scales not supported for TFLite post-processing')
      return postprocess.postprocess_tflite(self.config.as_dict(), cls_outputs,
                                            box_outputs)
    raise ValueError('Unsupported postprocess mode {}'.format(mode))

  def call(self, inputs, training=False, pre_mode='infer', post_mode='global'):
    """Call this model.

    Args:
      inputs: a tensor with common shape [batch, height, width, channels].
      training: If true, it is training mode. Otherwise, eval mode.
      pre_mode: preprocessing mode, must be {None, 'infer'}.
      post_mode: postprrocessing mode, must be {None, 'global', 'per_class'}.

    Returns:
      the output tensor list.
    """
    config = self.config

    # preprocess.
    inputs, scales = self._preprocessing(inputs, config.image_size,
                                         config.mean_rgb, config.stddev_rgb,
                                         pre_mode)
    # network.
    outputs = super().call(inputs, training)

    if 'object_detection' in config.heads and post_mode:
      # postprocess for detection
      det_outputs = self._postprocess(outputs[0], outputs[1], scales, post_mode)
      outputs = det_outputs + outputs[2:]

    return outputs
