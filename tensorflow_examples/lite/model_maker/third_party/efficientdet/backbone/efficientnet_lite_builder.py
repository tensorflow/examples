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
"""Model Builder for EfficientNet Edge Models.

efficientnet-litex (x=0,1,2,3,4) checkpoints are located in:
  https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/lite/efficientnet-litex.tar.gz
"""
import os
from absl import logging
import tensorflow.compat.v1 as tf

from tensorflow_examples.lite.model_maker.third_party.efficientdet import utils
from tensorflow_examples.lite.model_maker.third_party.efficientdet.backbone import efficientnet_builder
from tensorflow_examples.lite.model_maker.third_party.efficientdet.backbone import efficientnet_model

# Edge models use inception-style MEAN and STDDEV for better post-quantization.
MEAN_RGB = [127.0, 127.0, 127.0]
STDDEV_RGB = [128.0, 128.0, 128.0]


def efficientnet_lite_params(model_name):
  """Get efficientnet params based on model name."""
  params_dict = {
      # (width_coefficient, depth_coefficient, resolution, dropout_rate)
      'efficientnet-lite0': (1.0, 1.0, 224, 0.2),
      'efficientnet-lite1': (1.0, 1.1, 240, 0.2),
      'efficientnet-lite2': (1.1, 1.2, 260, 0.3),
      'efficientnet-lite3': (1.2, 1.4, 280, 0.3),
      'efficientnet-lite4': (1.4, 1.8, 300, 0.3),
  }
  return params_dict[model_name]


_DEFAULT_BLOCKS_ARGS = [
    'r1_k3_s11_e1_i32_o16_se0.25', 'r2_k3_s22_e6_i16_o24_se0.25',
    'r2_k5_s22_e6_i24_o40_se0.25', 'r3_k3_s22_e6_i40_o80_se0.25',
    'r3_k5_s11_e6_i80_o112_se0.25', 'r4_k5_s22_e6_i112_o192_se0.25',
    'r1_k3_s11_e6_i192_o320_se0.25',
]


def efficientnet_lite(width_coefficient=None,
                      depth_coefficient=None,
                      dropout_rate=0.2,
                      survival_prob=0.8):
  """Creates a efficientnet model."""
  global_params = efficientnet_model.GlobalParams(
      blocks_args=_DEFAULT_BLOCKS_ARGS,
      batch_norm_momentum=0.99,
      batch_norm_epsilon=1e-3,
      dropout_rate=dropout_rate,
      survival_prob=survival_prob,
      data_format='channels_last',
      num_classes=1000,
      width_coefficient=width_coefficient,
      depth_coefficient=depth_coefficient,
      depth_divisor=8,
      min_depth=None,
      relu_fn=tf.nn.relu6,  # Relu6 is for easier quantization.
      # The default is TPU-specific batch norm.
      # The alternative is tf.layers.BatchNormalization.
      batch_norm=utils.TpuBatchNormalization,  # TPU-specific requirement.
      clip_projection_output=False,
      fix_head_stem=True,  # Don't scale stem and head.
      local_pooling=True,  # special cases for tflite issues.
      use_se=False)  # SE is not well supported on many lite devices.
  return global_params


def get_model_params(model_name, override_params):
  """Get the block args and global params for a given model."""
  if model_name.startswith('efficientnet-lite'):
    width_coefficient, depth_coefficient, _, dropout_rate = (
        efficientnet_lite_params(model_name))
    global_params = efficientnet_lite(
        width_coefficient, depth_coefficient, dropout_rate)
  else:
    raise NotImplementedError('model name is not pre-defined: %s' % model_name)

  if override_params:
    # ValueError will be raised here if override_params has fields not included
    # in global_params.
    global_params = global_params._replace(**override_params)

  decoder = efficientnet_builder.BlockDecoder()
  blocks_args = decoder.decode(global_params.blocks_args)

  logging.info('global_params= %s', global_params)
  return blocks_args, global_params


def build_model(images,
                model_name,
                training,
                override_params=None,
                model_dir=None,
                fine_tuning=False,
                features_only=False,
                pooled_features_only=False):
  """A helper function to create a model and return predicted logits.

  Args:
    images: input images tensor.
    model_name: string, the predefined model name.
    training: boolean, whether the model is constructed for training.
    override_params: A dictionary of params for overriding. Fields must exist in
      efficientnet_model.GlobalParams.
    model_dir: string, optional model dir for saving configs.
    fine_tuning: boolean, whether the model is used for finetuning.
    features_only: build the base feature network only (excluding final
      1x1 conv layer, global pooling, dropout and fc head).
    pooled_features_only: build the base network for features extraction (after
      1x1 conv layer and global pooling, but before dropout and fc head).

  Returns:
    logits: the logits tensor of classes.
    endpoints: the endpoints for each layer.

  Raises:
    When model_name specified an undefined model, raises NotImplementedError.
    When override_params has invalid fields, raises ValueError.
  """
  assert isinstance(images, tf.Tensor)
  assert not (features_only and pooled_features_only)

  # For backward compatibility.
  if override_params and override_params.get('drop_connect_rate', None):
    override_params['survival_prob'] = 1 - override_params['drop_connect_rate']

  if not training or fine_tuning:
    if not override_params:
      override_params = {}
    override_params['batch_norm'] = utils.BatchNormalization
  blocks_args, global_params = get_model_params(model_name, override_params)

  if model_dir:
    param_file = os.path.join(model_dir, 'model_params.txt')
    if not tf.gfile.Exists(param_file):
      if not tf.gfile.Exists(model_dir):
        tf.gfile.MakeDirs(model_dir)
      with tf.gfile.GFile(param_file, 'w') as f:
        logging.info('writing to %s', param_file)
        f.write('model_name= %s\n\n' % model_name)
        f.write('global_params= %s\n\n' % str(global_params))
        f.write('blocks_args= %s\n\n' % str(blocks_args))

  model = efficientnet_model.Model(blocks_args, global_params, model_name)
  outputs = model(
      images,
      training=training,
      features_only=features_only,
      pooled_features_only=pooled_features_only)
  features, endpoints = outputs[0], outputs[1:]
  if features_only:
    features = tf.identity(features, 'features')
  elif pooled_features_only:
    features = tf.identity(features, 'pooled_features')
  else:
    features = tf.identity(features, 'logits')
  return features, endpoints


def build_model_base(images, model_name, training, override_params=None):
  """Create a base feature network and return the features before pooling.

  Args:
    images: input images tensor.
    model_name: string, the predefined model name.
    training: boolean, whether the model is constructed for training.
    override_params: A dictionary of params for overriding. Fields must exist in
      efficientnet_model.GlobalParams.

  Returns:
    features: base features before pooling.
    endpoints: the endpoints for each layer.

  Raises:
    When model_name specified an undefined model, raises NotImplementedError.
    When override_params has invalid fields, raises ValueError.
  """
  assert isinstance(images, tf.Tensor)
  # For backward compatibility.
  if override_params and override_params.get('drop_connect_rate', None):
    override_params['survival_prob'] = 1 - override_params['drop_connect_rate']

  blocks_args, global_params = get_model_params(model_name, override_params)

  model = efficientnet_model.Model(blocks_args, global_params, model_name)
  outputs = model(images, training=training, features_only=True)

  return outputs[0], outputs[1:]
