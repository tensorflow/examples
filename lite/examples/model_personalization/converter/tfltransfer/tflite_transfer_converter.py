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
"""TFLite converter for transfer learning models.

This converter is the first stage in the transfer learning pipeline.
It allows to convert a pair of models representing fixed
base and trainable head models to a set of TFLite models, which
can be then used by the transfer learning library.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
from tensorflow.compat import v1 as tfv1


class TFLiteTransferConverter(object):
  """Converter for transfer learning models.

  There are three parts of the input to the converter: base and
  head model configurations, and the optimizer configuration.
  Each of them has several variants, defined in the respective
  submodules, which are configured separately outside of the
  converter.

  The converter output format is currently a directory containing
  multiple TFLite models, but this should be considered an
  implementation detail and not relied upon.
  """

  def __init__(self, num_classes, base_model, head_model, optimizer,
               train_batch_size):
    """Creates a new converter instance.

    Args:
      num_classes: number of classes for the classification task.
      base_model: base model configuration of one of the supported types.
      head_model: head model configuration of one of the supported types.
      optimizer: optimizer configuration of one of the supported types.
      train_batch_size: batch size that will be used for training.
    """
    self.num_classes = num_classes
    self.base_model = base_model
    self.head_model = head_model
    self.optimizer = optimizer
    self.train_batch_size = train_batch_size

  def convert_and_save(self, out_model_dir):
    """Saves the converted model to a target directory."""
    if not os.path.isdir(out_model_dir):
      os.makedirs(out_model_dir)
    models = self._convert()

    for name, model in models.items():
      model_file_path = os.path.join(out_model_dir, name + '.tflite')
      with open(model_file_path, 'wb') as model_file:
        model_file.write(model)

  def _convert(self):
    """Converts all underlying models."""
    initialize_model_lite = self._generate_initialize_model()
    bottleneck_model_lite = self._generate_bottleneck_model()
    train_head_model_lite = self._generate_train_head_model()
    inference_model_lite = self._generate_inference_model()
    parameter_shapes = self._read_parameter_shapes(inference_model_lite)
    optimizer_model_lite = (
        self.optimizer.generate_optimizer_model(parameter_shapes))
    return {
        'initialize': initialize_model_lite,
        'bottleneck': bottleneck_model_lite,
        'train_head': train_head_model_lite,
        'inference': inference_model_lite,
        'optimizer': optimizer_model_lite,
    }

  def _read_parameter_shapes(self, inference_model):
    """Infers shapes of model parameters from the inference model."""
    interpreter = tfv1.lite.Interpreter(model_content=inference_model)
    return [
        parameter_in['shape'].tolist()
        for parameter_in in interpreter.get_input_details()[1:]
    ]

  def _generate_initialize_model(self):
    """Generates a model that outputs initial parameter values."""
    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [self.head_model.generate_initial_params().get_concrete_function()])
    return converter.convert()

  def _generate_bottleneck_model(self):
    """Converts the bottleneck model, i.e. the base model.

    Bottleneck is a name used in the transfer learning context for
    the base model outputs, which are at the same time head model
    inputs.

    Returns:
      TFLite bottleneck model.
    """
    return self.base_model.tflite_model()

  def _generate_train_head_model(self):
    """Converts the head training model.

    Head training model is constructed from the head model passed
    as converter input by adding a cross-entropy loss and gradient
    calculation for all variables in the input SavedModel.

    Returns:
      TFLite train head model.
    """
    with tf.Graph().as_default(), tfv1.Session() as sess:
      bottleneck_shape = ((self.train_batch_size,) +
                          self.head_model.input_shape())
      bottleneck = tfv1.placeholder(tf.float32, bottleneck_shape,
                                    'placeholder_bottleneck')

      # One-hot ground truth
      labels = tfv1.placeholder(tf.float32,
                                (self.train_batch_size, self.num_classes),
                                'placeholder_labels')

      loss, gradients, variables = self.head_model.train(bottleneck, labels)
      converter = tfv1.lite.TFLiteConverter.from_session(
          sess, [bottleneck, labels] + variables, [loss] + gradients)

      converter.allow_custom_ops = True
      if self.head_model.train_requires_flex():
        converter.target_ops = [tf.lite.OpsSet.SELECT_TF_OPS]
      return converter.convert()

  def _generate_inference_model(self):
    """Converts the head inference model.

    Inference model is constructed from the head model passed
    as converted input. It accepts as inputs the bottlenecks
    produces by the base model, and values for all trainable
    head model parameters.

    Returns:
      TFLite inference model.
    """
    with tf.Graph().as_default(), tfv1.Session() as sess:
      bottleneck_shape = ((1,) + self.head_model.input_shape())
      bottleneck = tfv1.placeholder(tf.float32, bottleneck_shape,
                                    'placeholder_bottleneck')
      predictions, head_variables = self.head_model.predict(bottleneck)
      converter = tfv1.lite.TFLiteConverter.from_session(
          sess, [bottleneck] + head_variables, [predictions])
      return converter.convert()
