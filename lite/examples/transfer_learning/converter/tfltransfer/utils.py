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
"""Helper utilities for various parts of the converter."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.compat import v1 as tfv1


def memoize(method):
  """A simple memoization decorator for zero-parameter methods."""

  # We use a class since Python 2 has no 'nonlocal'.
  class Memo(object):
    result = None

  def helper(self):
    if Memo.result is None:
      Memo.result = method(self)
    return Memo.result

  return helper


def convert_constants_to_placeholders(graph_def, constant_names):
  """Converts given constants in a GraphDef to placeholders."""
  constant_names = [tensor_to_op_name(name) for name in constant_names]
  output_graph_def = tfv1.GraphDef()
  for input_node in graph_def.node:
    output_node = tfv1.NodeDef()
    if input_node.name in constant_names:
      output_node.op = 'Placeholder'
      output_node.name = input_node.name
      output_node.attr['dtype'].CopyFrom(input_node.attr['dtype'])
      output_node.attr['shape'].shape.CopyFrom(
          input_node.attr['value'].tensor.tensor_shape)
    else:
      output_node.CopyFrom(input_node)
    output_graph_def.node.extend([output_node])

  output_graph_def.library.CopyFrom(graph_def.library)
  return output_graph_def


def tensor_to_op_name(tensor_name):
  """Strips tailing ':N' part from a tensor name.

  For example, 'dense/kernel:0', which is a tensor name, is converted
  to 'dense/kernel' which is the operation that outputs this tensor.

  Args:
    tensor_name: tensor name.

  Returns:
    Corresponding op name.
  """
  parts = tensor_name.split(':')
  if len(parts) == 1:
    return tensor_name
  assert len(parts) == 2
  return parts[0]
