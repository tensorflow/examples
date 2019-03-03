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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os

import tensorflow as tf
from keras import Model, Input
from keras.applications import MobileNet
from tensorflowjs.converters import load_keras_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelConverter:
  """
    Creates a ModelConverter class from a TensorFlow.js model file.

    Args: :param config_json_path: Full filepath of weights manifest file
    containing the model architecture. :param weights_path_prefix: Full filepath
    to the directory in which the weights binaries exist. :param
    tflite_model_file: Name of the TFLite FlatBuffer file to be exported.
    :return: ModelConverter class.
  """

  def __init__(self, config_json_path, weights_path_prefix, tflite_model_file):
    self.config_json_path = config_json_path
    self.weights_path_prefix = weights_path_prefix
    self.tflite_model_file = tflite_model_file
    self.keras_model_file = 'merged.h5'

    # MobileNet Options
    self.input_node_name = 'the_input'
    self.image_size = 224
    self.alpha = 0.25
    self.depth_multiplier = 1
    self._input_shape = (1, self.image_size, self.image_size, 3)
    self.depthwise_conv_layer = 'conv_pw_13_relu'

  def convert(self):
    self.save_keras_model()
    self._deserialize_tflite_from_keras()
    logger.info('The TFLite model has been generated')
    self._purge()

  def save_keras_model(self):
    top_model = load_keras_model(
        self.config_json_path,
        self.weights_path_prefix,
        weights_data_buffers=None,
        load_weights=True,
        use_unique_name_scope=True)

    base_model = self.get_base_model()
    merged_model = self.merge(base_model, top_model)
    merged_model.save(self.keras_model_file)

    logger.info('The merged Keras HDF5 model has been saved as {}'.format(
        self.keras_model_file))

  def merge(self, base_model, top_model):
    """
        Merges base model with the classification block
        :return:  Returns the merged Keras model
        """
    logger.info('Initializing model...')

    layer = base_model.get_layer(self.depthwise_conv_layer)
    model = Model(inputs=base_model.input, outputs=top_model(layer.output))
    logger.info('Model created.')

    return model

  def get_base_model(self):
    """
        Builds MobileNet with the default parameters
        :return:  Returns the base MobileNet model
        """
    input_tensor = Input(shape=self._input_shape[1:], name=self.input_node_name)
    base_model = MobileNet(
        input_shape=self._input_shape[1:],
        alpha=self.alpha,
        depth_multiplier=self.depth_multiplier,
        input_tensor=input_tensor,
        include_top=False)
    return base_model

  def _deserialize_tflite_from_keras(self):
    converter = tf.contrib.lite.TocoConverter.from_keras_model_file(
        self.keras_model_file)
    tflite_model = converter.convert()

    with open(self.tflite_model_file, 'wb') as file:
      file.write(tflite_model)

  def _purge(self):
    logger.info('Cleaning up Keras model')
    os.remove(self.keras_model_file)
