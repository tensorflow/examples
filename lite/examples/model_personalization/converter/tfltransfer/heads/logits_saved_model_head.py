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
"""Head model configuration for classifier SavedModels."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import tempfile

import tensorflow as tf
from tensorflow.compat import v1 as tfv1
from tensorflow.python.tools import freeze_graph

from tfltransfer import utils


class LogitsSavedModelHead(object):
  """Head model configuration for classifier SavedModels.

  This configuration supports input models that produce a logits
  tensor. Such models are to be trained on-device using cross-entropy
  loss applied to a softmax layer that is appended to logits.
  """

  def __init__(self,
               model_dir,
               tag=tf.saved_model.SERVING,
               signature_key='serving_default'):
    self.model_dir = model_dir
    self.tag = tag
    self.signature_key = signature_key

    # Pre-fetch some information about the model.
    loaded_model = tf.saved_model.load(model_dir, tags=[tag])
    self._signature = loaded_model.signatures[signature_key]

    input_def = next(self._signature.inputs.values().__iter__())
    self._input_shape = tuple(
        dim.size for dim in input_def.tensor_shape.dim[1:])

    variables = tfv1.global_variables()
    self._variable_names = [variable.name for variable in variables]
    self._initial_params = [variable.eval() for variable in variables]

    if len(self._signature.inputs) != 1:
      raise ValueError('Only single-input head models are supported')
    if len(self._signature.outputs) != 1:
      raise ValueError('Only single-output head models are supported')

  def predict(self, bottleneck, scope='head'):
    """Appends the serving signature of the model to the current graph.

    Bottleneck tensor is connected as an input to the added model.
    All model variables are converted to placeholders and returned
    in a list.

    Args:
      bottleneck: tensor in the current graph to be connected as input.
      scope: name of the scope to load the model into.

    Returns:
      (predictions tensor, list of variable placeholders)
    """
    logits, variables = self._logits(bottleneck, scope)
    predictions = tf.nn.softmax(logits)
    return predictions, variables

  def train(self, bottleneck, labels, scope='head'):
    """Appends the train signature of the model to the current graph.

    Bottleneck and labels tensors are connected as inputs.
    All model variables are converted to placeholders and returned
    in a list.

    Args:
      bottleneck: tensor containing input bottlenecks.
      labels: tensor containing ground truth labels.
      scope: name of the scope to load the model into.

    Returns:
      (loss tensor, list of variable gradients, list of variable placeholders)
    """
    logits, variables = self._logits(bottleneck, scope=scope)
    with tf.name_scope(scope + '/loss'):
      loss = tfv1.losses.softmax_cross_entropy(
          labels, logits, reduction=tfv1.losses.Reduction.SUM_OVER_BATCH_SIZE)
    with tf.name_scope(scope + '/backprop'):
      gradients = tf.gradients(loss, variables, stop_gradients=variables)
    return loss, gradients, variables

  def _logits(self, bottleneck, scope):
    """Appends the forward pass of the model."""
    input_name = (next(self._signature.inputs.values().__iter__()).name)
    output_name = (next(self._signature.outputs.values().__iter__()).name)
    output = tf.import_graph_def(
        self._frozen_graph_def(),
        name=scope,
        input_map={input_name: bottleneck},
        return_elements=[output_name])[0]
    variable_tensors = [
        tfv1.get_default_graph().get_tensor_by_name(scope + '/' + name)
        for name in self._variable_names
    ]
    return output, variable_tensors

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
      del zero  # Unused by this configuration since it does not use Fill.
      return [tf.constant(param) for param in self._initial_params]

    return model_func

  def input_shape(self):
    """Returns the model input shape."""
    return self._input_shape

  def train_requires_flex(self):
    """Whether the generated training model requires Flex support."""
    return True

  @utils.memoize
  def _frozen_graph_def(self):
    """Freezes the model and returns the frozen GraphDef.

    Frozen here means that all variables are converted to placeholders.

    Returns:
      Frozen GraphDef for the model.
    """
    temp_dir = tempfile.mkdtemp('tflite-transfer-convert')
    graph_def_file_name = os.path.join(temp_dir, 'frozen.pb')
    output_name = utils.tensor_to_op_name(
        next(self._signature.outputs.values().__iter__()).name)

    freeze_graph.freeze_graph(
        input_graph=None,
        input_saver=False,
        input_binary=True,
        input_checkpoint=None,
        output_node_names=output_name,
        restore_op_name=None,
        filename_tensor_name=None,
        output_graph=graph_def_file_name,
        clear_devices=True,
        initializer_nodes='',
        input_saved_model_dir=self.model_dir,
        saved_model_tags=self.tag)

    const_graph_def = tfv1.GraphDef()
    with open(graph_def_file_name, 'rb') as graph_def_file:
      const_graph_def.ParseFromString(graph_def_file.read())

    # Convert constants produced from variables to placeholders.
    graph_def = utils.convert_constants_to_placeholders(const_graph_def,
                                                        self._variable_names)

    shutil.rmtree(temp_dir)
    return graph_def
