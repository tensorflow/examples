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
"""Head model configuration for Keras models."""

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


class KerasModelHead(object):
  """Head model configuration for arbitrary Keras models.

  This configuration uses Keras-specific signatures to generate
  the transfer learning head model. Keras loss function that
  the model was compiled with is what the gradients will be
  computed on. Note that the optimizer used in Keras is not
  taken into account.
  """

  def __init__(self, keras_model):
    # Convert Keras model to SavedModel.
    saved_model_dir = tempfile.mkdtemp('tflite-transfer-keras-model')
    tf.keras.experimental.export_saved_model(keras_model, saved_model_dir)

    # Pre-fetch some information about the model.
    with tfv1.Session(graph=tf.Graph()) as sess:
      metagraph = tfv1.saved_model.load(sess, [tf.saved_model.SERVING],
                                        saved_model_dir)
      self._predict_signature = metagraph.signature_def.get('serving_default')

      input_def = next(self._predict_signature.inputs.values().__iter__())
      self._input_shape = tuple(
          dim.size for dim in input_def.tensor_shape.dim[1:])

      variables = tfv1.global_variables()
      self._variable_names = [variable.name for variable in variables]
      self._initial_params = [variable.eval() for variable in variables]
      trainable_variables = keras_model.trainable_variables
      self._trainable_variable_names = [
          variable.name for variable in trainable_variables
      ]

    with tfv1.Session(graph=tf.Graph()) as sess:
      eval_metagraph = tfv1.saved_model.load(sess, ['eval'], saved_model_dir)
      self._eval_signature = eval_metagraph.signature_def.get('eval')

    if len(self._predict_signature.inputs) != 1:
      raise ValueError('Only single-input head models are supported')
    if len(self._predict_signature.outputs) != 1:
      raise ValueError('Only single-output head models are supported')

    # Freeze the model.
    self._frozen_graph_def = self._freeze_keras_saved_model(saved_model_dir)

    # Clean up the temporary directory.
    shutil.rmtree(saved_model_dir)

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
    input_name = next(self._predict_signature.inputs.values().__iter__()).name
    output_name = (
        next(self._predict_signature.outputs.values().__iter__()).name)
    output = tf.import_graph_def(
        self._frozen_graph_def,
        name=scope,
        input_map={input_name: bottleneck},
        return_elements=[output_name])[0]
    variable_tensors = [
        tfv1.get_default_graph().get_tensor_by_name(scope + '/' + name)
        for name in self._variable_names
    ]
    return output, variable_tensors

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

    Raises:
      RuntimeError: if model signature does not conform to expectations.
    """
    bottleneck_names = [
        input_def.name
        for key, input_def in self._eval_signature.inputs.items()
        if key.endswith('_input') or key.startswith('input_')
    ]
    labels_names = [
        input_def.name
        for key, input_def in self._eval_signature.inputs.items()
        if key.endswith('_target') or key.startswith('target_')
    ]
    if len(bottleneck_names) != 1 or len(labels_names) != 1:
      raise RuntimeError('Unexpected Keras eval signature inputs')
    bottleneck_name = bottleneck_names[0]
    labels_name = labels_names[0]
    loss_name = self._eval_signature.outputs['loss'].name
    input_map = {
        bottleneck_name: bottleneck,
        labels_name: labels,
    }
    loss = tf.import_graph_def(
        self._frozen_graph_def,
        name=scope,
        input_map=input_map,
        return_elements=[loss_name])[0]
    train_variables = [
        tfv1.get_default_graph().get_tensor_by_name(scope + '/' + name)
        for name in self._trainable_variable_names
    ]
    variables = [
        tfv1.get_default_graph().get_tensor_by_name(scope + '/' + name)
        for name in self._variable_names
    ]
    with tf.name_scope(scope + '/backprop'):
      gradients = tf.gradients(
          loss, train_variables, stop_gradients=train_variables)
    return loss, gradients, variables

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

  def _freeze_keras_saved_model(self, saved_model_dir):
    """Freezes the model and returns the frozen GraphDef.

    Frozen here means that all variables are converted to placeholders.

    Args:
      saved_model_dir: Directory with the Keras SavedModel export.

    Returns:
      Frozen GraphDef for the model.
    """
    temp_dir = tempfile.mkdtemp('tflite-transfer-convert')
    graph_def_file_name = os.path.join(temp_dir, 'frozen.pb')
    output_names = [
        utils.tensor_to_op_name(output.name)
        for output in self._eval_signature.outputs.values()
    ]

    freeze_graph.freeze_graph(
        input_graph=None,
        input_saver=False,
        input_binary=True,
        input_checkpoint=None,
        output_node_names=','.join(output_names),
        restore_op_name=None,
        filename_tensor_name=None,
        output_graph=graph_def_file_name,
        clear_devices=True,
        initializer_nodes='',
        input_saved_model_dir=saved_model_dir,
        saved_model_tags='eval')

    const_graph_def = tfv1.GraphDef()
    with open(graph_def_file_name, 'rb') as graph_def_file:
      const_graph_def.ParseFromString(graph_def_file.read())

    # Convert constants produced from trainable variables to placeholders.
    # Note: eval model might have other variables that should not be trainable,
    # they are kept as constants. Only variables that are present in serve
    # model are converted.
    graph_def = utils.convert_constants_to_placeholders(const_graph_def,
                                                        self._variable_names)

    shutil.rmtree(temp_dir)
    return graph_def
