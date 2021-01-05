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
"""Backbone network factory."""
import os
from absl import logging
import tensorflow as tf

from tensorflow_examples.lite.model_maker.third_party.efficientdet.backbone import efficientnet_builder
from tensorflow_examples.lite.model_maker.third_party.efficientdet.backbone import efficientnet_lite_builder
from tensorflow_examples.lite.model_maker.third_party.efficientdet.backbone import efficientnet_model


def get_model_builder(model_name):
  """Get the model_builder module for a given model name."""
  if model_name.startswith('efficientnet-lite'):
    return efficientnet_lite_builder
  elif model_name.startswith('efficientnet-'):
    return efficientnet_builder
  else:
    raise ValueError('Unknown model name {}'.format(model_name))


def get_model(model_name, override_params=None, model_dir=None):
  """A helper function to create and return model.

  Args:
    model_name: string, the predefined model name.
    override_params: A dictionary of params for overriding. Fields must exist in
      efficientnet_model.GlobalParams.
    model_dir: string, optional model dir for saving configs.

  Returns:
    created model

  Raises:
    When model_name specified an undefined model, raises NotImplementedError.
    When override_params has invalid fields, raises ValueError.
  """

  # For backward compatibility.
  if override_params and override_params.get('drop_connect_rate', None):
    override_params['survival_prob'] = 1 - override_params['drop_connect_rate']

  if not override_params:
    override_params = {}

  if model_name.startswith('efficientnet-lite'):
    builder = efficientnet_lite_builder
  elif model_name.startswith('efficientnet-'):
    builder = efficientnet_builder
  else:
    raise ValueError('Unknown model name {}'.format(model_name))

  blocks_args, global_params = builder.get_model_params(model_name,
                                                        override_params)

  if model_dir:
    param_file = os.path.join(model_dir, 'model_params.txt')
    if not tf.io.gfile.exists(param_file):
      if not tf.io.gfile.exists(model_dir):
        tf.io.gfile.mkdir(model_dir)
      with tf.io.gfile.GFile(param_file, 'w') as f:
        logging.info('writing to %s', param_file)
        f.write('model_name= %s\n\n' % model_name)
        f.write('global_params= %s\n\n' % str(global_params))
        f.write('blocks_args= %s\n\n' % str(blocks_args))

  return efficientnet_model.Model(blocks_args, global_params, model_name)
