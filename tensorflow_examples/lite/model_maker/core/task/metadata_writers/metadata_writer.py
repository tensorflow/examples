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
# ==============================================================================
"""Helper class to write metadata into TFLite models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import os

import tensorflow as tf

# pylint: disable=g-direct-tensorflow-import
from tflite_support import metadata as _metadata
from tflite_support import schema_py_generated as _schema_fb
# pylint: enable=g-direct-tensorflow-import


class MetadataWriter(abc.ABC):
  """Writes the metadata and associated file into a TFLite model."""

  def __init__(self, model_file, export_directory, associated_files):
    """Constructs the MetadataWriter.

    Args:
      model_file: the path to the model to be populated.
      export_directory: path to the directory where the model and json file will
        be exported to.
      associated_files: path to the associated files to be populated.
    """
    self.model_file = model_file
    self.export_directory = export_directory
    self.associated_files = associated_files
    self.metadata_buf = None

  def populate(self):
    """Creates metadata and then populates it into a TFLite model."""
    self._create_metadata()
    self._populate_metadata()

  @abc.abstractmethod
  def _create_metadata(self):
    pass

  def _get_subgraph(self):
    """Gets the input / output tensor names into lists."""

    with open(self.model_file, "rb") as f:
      model_buf = f.read()

    model = _schema_fb.Model.GetRootAsModel(model_buf, 0)
    # There should be exactly one SubGraph in the model.
    if model.SubgraphsLength() != 1:
      raise ValueError(
          "The model should have exactly one subgraph, but found {0}.".format(
              model.SubgraphsLength()))

    return model.Subgraphs(0)

  def _get_input_tensor_names(self):
    """Gets the input tensor names into a list."""
    input_names = []
    subgraph = self._get_subgraph()
    for i in range(subgraph.InputsLength()):
      index = subgraph.Inputs(i)
      input_names.append(subgraph.Tensors(index).Name())

    return input_names

  def _get_output_tensor_names(self):
    """Gets the output tensor names into a list."""
    output_names = []
    subgraph = self._get_subgraph()
    for i in range(subgraph.OutputsLength()):
      index = subgraph.Outputs(i)
      output_names.append(subgraph.Tensors(index).Name())

    return output_names

  def _order_tensor_metadata_with_names(self, tensor_metadata, tensor_names,
                                        ordered_tensor_names):
    """Orders the tensor metadata array according to the ordered tensor names.

    Args:
      tensor_metadata: the tensor_metadata array in list.
      tensor_names: name list of the tensors corresponding to tensor_metadata.
      ordered_tensor_names: name list of the tensors in the expected order, such
        as in the same order as saved in the model.

    Returns:
      The ordered tensor metadata list.
    """
    if len(tensor_names) != len(tensor_metadata):
      raise ValueError((
          "Number of the tensor names ({0}) does not match the number of tensor"
          " metadata ({1}).".format(len(tensor_names), len(tensor_metadata))))
    if len(ordered_tensor_names) != len(tensor_names):
      raise ValueError(
          ("Number of the ordered tensor names ({0}) does not match the number "
           "of tensor names ({1}).".format(
               len(ordered_tensor_names), len(tensor_names))))

    ordered_metadata = []
    name_meta_dict = dict(zip(tensor_names, tensor_metadata))
    for name in ordered_tensor_names:
      ordered_metadata.append(name_meta_dict[name.decode()])
    return ordered_metadata

  def _populate_metadata(self):
    """Populates the metadata and label file to the model file."""
    # Copies model_file to export_path.
    model_basename = os.path.basename(self.model_file)
    export_model_path = os.path.join(self.export_directory, model_basename)
    if os.path.abspath(self.model_file) != os.path.abspath(export_model_path):
      tf.io.gfile.copy(self.model_file, export_model_path, overwrite=True)

    populator = _metadata.MetadataPopulator.with_model_file(export_model_path)
    populator.load_metadata_buffer(self.metadata_buf)
    if self.associated_files:
      populator.load_associated_files(self.associated_files)
    populator.populate()

    # Displays the model metadata.
    displayer = _metadata.MetadataDisplayer.with_model_file(export_model_path)
    export_json_path = os.path.join(
        self.export_directory,
        os.path.splitext(model_basename)[0] + ".json")
    with open(export_json_path, "w") as f:
      f.write(displayer.get_metadata_json())

    print("Finished populating metadata and associated file to the model:")
    print(export_model_path)
    print("The metadata json file has been saved to:")
    print(export_json_path)
    if self.associated_files:
      print("The associated file that has been been packed to the model is:")
      print(displayer.get_packed_associated_file_list())
