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
"""EfficientDet model definition.

[1] Mingxing Tan, Ruoming Pang, Quoc Le.
    EfficientDet: Scalable and Efficient Object Detection.
    CVPR 2020, https://arxiv.org/abs/1911.09070
"""
import functools
import re

from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf

from tensorflow_examples.lite.model_maker.third_party.efficientdet import hparams_config
from tensorflow_examples.lite.model_maker.third_party.efficientdet import utils
from tensorflow_examples.lite.model_maker.third_party.efficientdet.backbone import backbone_factory
from tensorflow_examples.lite.model_maker.third_party.efficientdet.backbone import efficientnet_builder
from tensorflow_examples.lite.model_maker.third_party.efficientdet.keras import fpn_configs


################################################################################
def freeze_vars(variables, pattern):
  """Removes backbone+fpn variables from the input.

  Args:
    variables: all the variables in training
    pattern: a reg experession such as ".*(efficientnet|fpn_cells).*".

  Returns:
    var_list: a list containing variables for training
  """
  if pattern:
    filtered_vars = [v for v in variables if not re.match(pattern, v.name)]
    if len(filtered_vars) == len(variables):
      logging.warning('%s didnt match with any variable. Please use compatible '
                      'pattern. i.e "(efficientnet)"', pattern)
    return filtered_vars
  return variables


def resample_feature_map(feat,
                         name,
                         target_height,
                         target_width,
                         target_num_channels,
                         apply_bn=False,
                         is_training=None,
                         conv_after_downsample=False,
                         strategy=None,
                         data_format='channels_last',
                         batch_norm_trainable=True):
  """Resample input feature map to have target number of channels and size."""
  if data_format == 'channels_first':
    _, num_channels, height, width = feat.get_shape().as_list()
  else:
    _, height, width, num_channels = feat.get_shape().as_list()

  if height is None or width is None or num_channels is None:
    raise ValueError(
        'shape[1] or shape[2] or shape[3] of feat is None (shape:{}).'.format(
            feat.shape))
  if apply_bn and is_training is None:
    raise ValueError('If BN is applied, need to provide is_training')

  def _maybe_apply_1x1(feat):
    """Apply 1x1 conv to change layer width if necessary."""
    if num_channels != target_num_channels:
      feat = tf.layers.conv2d(
          feat,
          filters=target_num_channels,
          kernel_size=(1, 1),
          padding='same',
          data_format=data_format)
      if apply_bn:
        feat = utils.batch_norm_act(
            feat,
            is_training_bn=is_training,
            act_type=None,
            data_format=data_format,
            strategy=strategy,
            batch_norm_trainable=batch_norm_trainable,
            name='bn')
    return feat

  with tf.variable_scope('resample_{}'.format(name)):
    # If conv_after_downsample is True, when downsampling, apply 1x1 after
    # downsampling for efficiency.
    if height > target_height and width > target_width:
      if not conv_after_downsample:
        feat = _maybe_apply_1x1(feat)
      height_stride_size = int((height - 1) // target_height + 1)
      width_stride_size = int((width - 1) // target_width + 1)

      # Use max pooling in default.
      feat = tf.layers.max_pooling2d(
          inputs=feat,
          pool_size=[height_stride_size + 1, width_stride_size + 1],
          strides=[height_stride_size, width_stride_size],
          padding='SAME',
          data_format=data_format)

      if conv_after_downsample:
        feat = _maybe_apply_1x1(feat)
    elif height <= target_height and width <= target_width:
      feat = _maybe_apply_1x1(feat)
      if height < target_height or width < target_width:
        if data_format == 'channels_first':
          feat = tf.transpose(feat, [0, 2, 3, 1])
        feat = tf.cast(
            tf.image.resize_nearest_neighbor(
                tf.cast(feat, tf.float32), [target_height, target_width]),
            dtype=feat.dtype)
        if data_format == 'channels_first':
          feat = tf.transpose(feat, [0, 3, 1, 2])
    else:
      raise ValueError(
          'Incompatible target feature map size: target_height: {},'
          'target_width: {}'.format(target_height, target_width))

  return feat


###############################################################################
def class_net(images,
              level,
              num_classes,
              num_anchors,
              num_filters,
              is_training,
              act_type,
              separable_conv=True,
              repeats=4,
              survival_prob=None,
              strategy=None,
              data_format='channels_last'):
  """Class prediction network."""
  if separable_conv:
    conv_op = functools.partial(
        tf.layers.separable_conv2d, depth_multiplier=1,
        data_format=data_format,
        pointwise_initializer=tf.initializers.variance_scaling(),
        depthwise_initializer=tf.initializers.variance_scaling())
  else:
    conv_op = functools.partial(
        tf.layers.conv2d,
        data_format=data_format,
        kernel_initializer=tf.random_normal_initializer(stddev=0.01))

  for i in range(repeats):
    orig_images = images
    images = conv_op(
        images,
        num_filters,
        kernel_size=3,
        bias_initializer=tf.zeros_initializer(),
        activation=None,
        padding='same',
        name='class-%d' % i)
    images = utils.batch_norm_act(
        images,
        is_training,
        act_type=act_type,
        init_zero=False,
        strategy=strategy,
        data_format=data_format,
        name='class-%d-bn-%d' % (i, level))

    if i > 0 and survival_prob:
      images = utils.drop_connect(images, is_training, survival_prob)
      images = images + orig_images

  classes = conv_op(
      images,
      num_classes * num_anchors,
      kernel_size=3,
      bias_initializer=tf.constant_initializer(-np.log((1 - 0.01) / 0.01)),
      padding='same',
      name='class-predict')
  return classes


def box_net(images,
            level,
            num_anchors,
            num_filters,
            is_training,
            act_type,
            repeats=4,
            separable_conv=True,
            survival_prob=None,
            strategy=None,
            data_format='channels_last'):
  """Box regression network."""
  if separable_conv:
    conv_op = functools.partial(
        tf.layers.separable_conv2d, depth_multiplier=1,
        data_format=data_format,
        pointwise_initializer=tf.initializers.variance_scaling(),
        depthwise_initializer=tf.initializers.variance_scaling())
  else:
    conv_op = functools.partial(
        tf.layers.conv2d,
        data_format=data_format,
        kernel_initializer=tf.random_normal_initializer(stddev=0.01))

  for i in range(repeats):
    orig_images = images
    images = conv_op(
        images,
        num_filters,
        kernel_size=3,
        activation=None,
        bias_initializer=tf.zeros_initializer(),
        padding='same',
        name='box-%d' % i)
    images = utils.batch_norm_act(
        images,
        is_training,
        act_type=act_type,
        init_zero=False,
        strategy=strategy,
        data_format=data_format,
        name='box-%d-bn-%d' % (i, level))

    if i > 0 and survival_prob:
      images = utils.drop_connect(images, is_training, survival_prob)
      images = images + orig_images

  boxes = conv_op(
      images,
      4 * num_anchors,
      kernel_size=3,
      bias_initializer=tf.zeros_initializer(),
      padding='same',
      name='box-predict')

  return boxes


def build_class_and_box_outputs(feats, config):
  """Builds box net and class net.

  Args:
   feats: input tensor.
   config: a dict-like config, including all parameters.

  Returns:
   A tuple (class_outputs, box_outputs) for class/box predictions.
  """

  class_outputs = {}
  box_outputs = {}
  num_anchors = len(config.aspect_ratios) * config.num_scales
  cls_fsize = config.fpn_num_filters
  with tf.variable_scope('class_net', reuse=tf.AUTO_REUSE):
    for level in range(config.min_level,
                       config.max_level + 1):
      class_outputs[level] = class_net(
          images=feats[level],
          level=level,
          num_classes=config.num_classes,
          num_anchors=num_anchors,
          num_filters=cls_fsize,
          is_training=config.is_training_bn,
          act_type=config.act_type,
          repeats=config.box_class_repeats,
          separable_conv=config.separable_conv,
          survival_prob=config.survival_prob,
          strategy=config.strategy,
          data_format=config.data_format
          )

  box_fsize = config.fpn_num_filters
  with tf.variable_scope('box_net', reuse=tf.AUTO_REUSE):
    for level in range(config.min_level,
                       config.max_level + 1):
      box_outputs[level] = box_net(
          images=feats[level],
          level=level,
          num_anchors=num_anchors,
          num_filters=box_fsize,
          is_training=config.is_training_bn,
          act_type=config.act_type,
          repeats=config.box_class_repeats,
          separable_conv=config.separable_conv,
          survival_prob=config.survival_prob,
          strategy=config.strategy,
          data_format=config.data_format)

  return class_outputs, box_outputs


def build_backbone(features, config):
  """Builds backbone model.

  Args:
   features: input tensor.
   config: config for backbone, such as is_training_bn and backbone name.

  Returns:
    A dict from levels to the feature maps from the output of the backbone model
    with strides of 8, 16 and 32.

  Raises:
    ValueError: if backbone_name is not supported.
  """
  backbone_name = config.backbone_name
  is_training_bn = config.is_training_bn
  if 'efficientnet' in backbone_name:
    override_params = {
        'batch_norm':
            utils.batch_norm_class(is_training_bn, config.strategy),
        'relu_fn':
            functools.partial(utils.activation_fn, act_type=config.act_type),
    }
    if 'b0' in backbone_name:
      override_params['survival_prob'] = 0.0
    if config.backbone_config is not None:
      override_params['blocks_args'] = (
          efficientnet_builder.BlockDecoder().encode(
              config.backbone_config.blocks))
    override_params['data_format'] = config.data_format
    model_builder = backbone_factory.get_model_builder(backbone_name)
    _, endpoints = model_builder.build_model_base(
        features,
        backbone_name,
        training=is_training_bn,
        override_params=override_params)
    u1 = endpoints[0]
    u2 = endpoints[1]
    u3 = endpoints[2]
    u4 = endpoints[3]
    u5 = endpoints[4]
  else:
    raise ValueError(
        'backbone model {} is not supported.'.format(backbone_name))
  return {0: features, 1: u1, 2: u2, 3: u3, 4: u4, 5: u5}


def build_feature_network(features, config):
  """Build FPN input features.

  Args:
   features: input tensor.
   config: a dict-like config, including all parameters.

  Returns:
    A dict from levels to the feature maps processed after feature network.
  """
  feat_sizes = utils.get_feat_sizes(config.image_size, config.max_level)
  feats = []
  if config.min_level not in features.keys():
    raise ValueError('features.keys ({}) should include min_level ({})'.format(
        features.keys(), config.min_level))

  # Build additional input features that are not from backbone.
  for level in range(config.min_level, config.max_level + 1):
    if level in features.keys():
      feats.append(features[level])
    else:
      h_id, w_id = (2, 3) if config.data_format == 'channels_first' else (1, 2)
      # Adds a coarser level by downsampling the last feature map.
      feats.append(
          resample_feature_map(
              feats[-1],
              name='p%d' % level,
              target_height=(feats[-1].shape[h_id] - 1) // 2 + 1,
              target_width=(feats[-1].shape[w_id] - 1) // 2 + 1,
              target_num_channels=config.fpn_num_filters,
              apply_bn=config.apply_bn_for_resampling,
              is_training=config.is_training_bn,
              conv_after_downsample=config.conv_after_downsample,
              strategy=config.strategy,
              data_format=config.data_format,
              batch_norm_trainable=config.batch_norm_trainable
          ))

  utils.verify_feats_size(
      feats,
      feat_sizes=feat_sizes,
      min_level=config.min_level,
      max_level=config.max_level,
      data_format=config.data_format)

  with tf.variable_scope('fpn_cells'):
    for rep in range(config.fpn_cell_repeats):
      with tf.variable_scope('cell_{}'.format(rep)):
        logging.info('building cell %d', rep)
        new_feats = build_bifpn_layer(feats, feat_sizes, config)

        feats = [
            new_feats[level]
            for level in range(
                config.min_level, config.max_level + 1)
        ]

        utils.verify_feats_size(
            feats,
            feat_sizes=feat_sizes,
            min_level=config.min_level,
            max_level=config.max_level,
            data_format=config.data_format)

  return new_feats


def fuse_features(nodes, weight_method):
  """Fuse features from different resolutions and return a weighted sum.

  Args:
    nodes: a list of tensorflow features at different levels
    weight_method: feature fusion method. One of:
      - "attn" - Softmax weighted fusion
      - "fastattn" - Fast normalzied feature fusion
      - "sum" - a sum of inputs

  Returns:
    A tensor denoting the fused feature.
  """
  dtype = nodes[0].dtype

  if weight_method == 'attn':
    edge_weights = [tf.cast(tf.Variable(1.0, name='WSM'), dtype=dtype)
                    for _ in nodes]
    normalized_weights = tf.nn.softmax(tf.stack(edge_weights))
    nodes = tf.stack(nodes, axis=-1)
    new_node = tf.reduce_sum(nodes * normalized_weights, -1)
  elif weight_method == 'fastattn':
    edge_weights = []
    for i, _ in enumerate(nodes):
      edge_weights.append(
          tf.nn.relu(
              tf.get_variable(
                  'WSM' if i == 0 else f'WSM_{i}',
                  shape=[],
                  initializer=tf.ones_initializer(),
                  dtype=dtype)))

    weights_sum = tf.add_n(edge_weights)
    nodes = [nodes[i] * edge_weights[i] / (weights_sum + 0.0001)
             for i in range(len(nodes))]
    new_node = tf.add_n(nodes)
  elif weight_method == 'channel_attn':
    num_filters = int(nodes[0].shape[-1])
    edge_weights = [
        tf.cast(
            tf.Variable(lambda: tf.ones([num_filters]), name='WSM'),
            dtype=dtype) for _ in nodes
    ]
    normalized_weights = tf.nn.softmax(tf.stack(edge_weights, -1), axis=-1)
    nodes = tf.stack(nodes, axis=-1)
    new_node = tf.reduce_sum(nodes * normalized_weights, -1)
  elif weight_method == 'channel_fastattn':
    num_filters = int(nodes[0].shape[-1])
    edge_weights = [
        tf.nn.relu(tf.cast(
            tf.Variable(lambda: tf.ones([num_filters]), name='WSM'),
            dtype=dtype)) for _ in nodes
    ]
    weights_sum = tf.add_n(edge_weights)
    nodes = [nodes[i] * edge_weights[i] / (weights_sum + 0.0001)
             for i in range(len(nodes))]
    new_node = tf.add_n(nodes)
  elif weight_method == 'sum':
    new_node = tf.add_n(nodes)
  else:
    raise ValueError(
        'unknown weight_method {}'.format(weight_method))

  return new_node


def build_bifpn_layer(feats, feat_sizes, config):
  """Builds a feature pyramid given previous feature pyramid and config."""
  p = config  # use p to denote the network config.
  if p.fpn_config:
    fpn_config = p.fpn_config
  else:
    fpn_config = fpn_configs.get_fpn_config(p.fpn_name, p.min_level,
                                            p.max_level, p.fpn_weight_method)

  num_output_connections = [0 for _ in feats]
  for i, fnode in enumerate(fpn_config.nodes):
    with tf.variable_scope('fnode{}'.format(i)):
      logging.info('fnode %d : %s', i, fnode)
      new_node_height = feat_sizes[fnode['feat_level']]['height']
      new_node_width = feat_sizes[fnode['feat_level']]['width']
      nodes = []
      for idx, input_offset in enumerate(fnode['inputs_offsets']):
        input_node = feats[input_offset]
        num_output_connections[input_offset] += 1
        input_node = resample_feature_map(
            input_node, '{}_{}_{}'.format(idx, input_offset, len(feats)),
            new_node_height, new_node_width, p.fpn_num_filters,
            p.apply_bn_for_resampling, p.is_training_bn,
            p.conv_after_downsample,
            strategy=p.strategy,
            data_format=config.data_format,
            batch_norm_trainable=p.batch_norm_trainable)
        nodes.append(input_node)

      new_node = fuse_features(nodes, fpn_config.weight_method)

      with tf.variable_scope('op_after_combine{}'.format(len(feats))):
        if not p.conv_bn_act_pattern:
          new_node = utils.activation_fn(new_node, p.act_type)

        if p.separable_conv:
          conv_op = functools.partial(
              tf.layers.separable_conv2d, depth_multiplier=1)
        else:
          conv_op = tf.layers.conv2d

        new_node = conv_op(
            new_node,
            filters=p.fpn_num_filters,
            kernel_size=(3, 3),
            padding='same',
            use_bias=not p.conv_bn_act_pattern,
            data_format=config.data_format,
            name='conv')

        new_node = utils.batch_norm_act(
            new_node,
            is_training_bn=p.is_training_bn,
            act_type=None if not p.conv_bn_act_pattern else p.act_type,
            data_format=config.data_format,
            strategy=p.strategy,
            batch_norm_trainable=p.batch_norm_trainable,
            name='bn')

      feats.append(new_node)
      num_output_connections.append(0)

  output_feats = {}
  for l in range(p.min_level, p.max_level + 1):
    for i, fnode in enumerate(reversed(fpn_config.nodes)):
      if fnode['feat_level'] == l:
        output_feats[l] = feats[-1 - i]
        break
  return output_feats


def efficientdet(features, model_name=None, config=None, **kwargs):
  """Build EfficientDet model."""
  if not config and not model_name:
    raise ValueError('please specify either model name or config')

  if not config:
    config = hparams_config.get_efficientdet_config(model_name)
  elif isinstance(config, dict):
    config = hparams_config.Config(config)  # wrap dict in Config object

  if kwargs:
    config.override(kwargs)

  logging.info(config)

  # build backbone features.
  features = build_backbone(features, config)
  logging.info('backbone params/flops = {:.6f}M, {:.9f}B'.format(
      *utils.num_params_flops()))

  # build feature network.
  fpn_feats = build_feature_network(features, config)
  logging.info('backbone+fpn params/flops = {:.6f}M, {:.9f}B'.format(
      *utils.num_params_flops()))

  # build class and box predictions.
  class_outputs, box_outputs = build_class_and_box_outputs(fpn_feats, config)
  logging.info('backbone+fpn+box params/flops = {:.6f}M, {:.9f}B'.format(
      *utils.num_params_flops()))

  return class_outputs, box_outputs
