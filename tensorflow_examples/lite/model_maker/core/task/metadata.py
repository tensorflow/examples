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
"""Metadata populator for TFLite models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
from tensorflow_examples.lite.model_maker.core.task import model_spec as ms

TFLITE_SUPPORT_TOOLS_INSTALLED = True

try:
  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top
  from tflite_support import flatbuffers
  from tflite_support import metadata as _metadata
  from tflite_support import metadata_schema_py_generated as _metadata_fb
  # pylint: enable=g-direct-tensorflow-import,g-import-not-at-top
except ImportError:
  tf.compat.v1.logging.warning("Needs to install tflite-support package.")
  TFLITE_SUPPORT_TOOLS_INSTALLED = False


def export_metadata_json_file(tflite_file):
  """Exports metadata to json file."""
  displayer = _metadata.MetadataDisplayer.with_model_file(tflite_file)
  export_directory = os.path.dirname(tflite_file)
  try:
    json_file = os.path.join(
        export_directory,
        os.path.splitext(os.path.basename(tflite_file))[0] + ".json")
    with open(json_file, "w") as f:
      content = displayer.get_metadata_json()
      f.write(content)
  except AttributeError:
    # TODO(yuqili): Remove this line once the API is stable.
    displayer.export_metadata_json_file(export_directory)


class ImageModelSpecificInfo(object):
  """Holds information that is specificly tied to an image classifier."""

  def __init__(self,
               name,
               version,
               image_width,
               image_height,
               mean,
               std,
               image_min=0,
               image_max=1):
    self.name = name
    self.version = version
    self.image_width = image_width
    self.image_height = image_height
    self.mean = mean
    self.std = std
    self.image_min = image_min
    self.image_max = image_max


def get_model_info(model_spec, quantized=False, version="v1"):
  if not isinstance(model_spec, ms.ImageModelSpec):
    raise ValueError("Currently only support models for image classification.")

  name = model_spec.name
  if quantized:
    name.append("_quantized")
  return ImageModelSpecificInfo(
      model_spec.name,
      version,
      image_width=model_spec.input_image_shape[1],
      image_height=model_spec.input_image_shape[0],
      mean=model_spec.mean_rgb,
      std=model_spec.stddev_rgb)


class MetadataPopulatorForImageClassifier(object):
  """Populates the metadata for an image classifier."""

  def __init__(self, model_file, model_info, label_file_path):
    self.model_file = model_file
    self.model_info = model_info
    self.label_file_path = label_file_path
    self.metadata_buf = None

  def populate(self):
    """Creates metadata and then populates it for an image classifier."""
    self._create_metadata()
    self._populate_metadata()

  def _create_metadata(self):
    """Creates the metadata for an image classifier."""

    # Creates model info.
    model_meta = _metadata_fb.ModelMetadataT()
    model_meta.name = self.model_info.name
    model_meta.description = ("Identify the most prominent object in the "
                              "image from a set of categories.")
    model_meta.version = self.model_info.version
    model_meta.author = "TFLite Model Maker"
    model_meta.license = ("Apache License. Version 2.0 "
                          "http://www.apache.org/licenses/LICENSE-2.0.")

    # Creates input info.
    input_meta = _metadata_fb.TensorMetadataT()
    input_meta.name = "image"
    input_meta.description = (
        "Input image to be classified. The expected image is {0} x {1}, with "
        "three channels (red, blue, and green) per pixel. Each value in the "
        "tensor is a single byte between {2} and {3}.".format(
            self.model_info.image_width, self.model_info.image_height,
            self.model_info.image_min, self.model_info.image_max))
    input_meta.content = _metadata_fb.ContentT()
    input_meta.content.contentProperties = _metadata_fb.ImagePropertiesT()
    input_meta.content.contentProperties.colorSpace = (
        _metadata_fb.ColorSpaceType.RGB)
    input_meta.content.contentPropertiesType = (
        _metadata_fb.ContentProperties.ImageProperties)
    input_normalization = _metadata_fb.ProcessUnitT()
    input_normalization.optionsType = (
        _metadata_fb.ProcessUnitOptions.NormalizationOptions)
    input_normalization.options = _metadata_fb.NormalizationOptionsT()
    input_normalization.options.mean = self.model_info.mean
    input_normalization.options.std = self.model_info.std
    input_meta.processUnits = [input_normalization]
    input_stats = _metadata_fb.StatsT()
    input_stats.max = [self.model_info.image_max]
    input_stats.min = [self.model_info.image_min]
    input_meta.stats = input_stats

    # Creates output info.
    output_meta = _metadata_fb.TensorMetadataT()
    output_meta.name = "probability"
    output_meta.description = "Probabilities of the labels respectively."
    output_meta.content = _metadata_fb.ContentT()
    output_meta.content.content_properties = _metadata_fb.FeaturePropertiesT()
    output_meta.content.contentPropertiesType = (
        _metadata_fb.ContentProperties.FeatureProperties)
    output_stats = _metadata_fb.StatsT()
    output_stats.max = [1.0]
    output_stats.min = [0.0]
    output_meta.stats = output_stats
    label_file = _metadata_fb.AssociatedFileT()
    label_file.name = os.path.basename(self.label_file_path)
    label_file.description = "Labels that %s can recognize." % model_meta.name
    label_file.type = _metadata_fb.AssociatedFileType.TENSOR_AXIS_LABELS
    output_meta.associatedFiles = [label_file]

    # Creates subgraph info.
    subgraph = _metadata_fb.SubGraphMetadataT()
    subgraph.inputTensorMetadata = [input_meta]
    subgraph.outputTensorMetadata = [output_meta]
    model_meta.subgraphMetadata = [subgraph]

    b = flatbuffers.Builder(0)
    b.Finish(
        model_meta.Pack(b),
        _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER)
    self.metadata_buf = b.Output()

  def _populate_metadata(self):
    """Populates metadata and label file to the model file."""
    populator = _metadata.MetadataPopulator.with_model_file(self.model_file)
    populator.load_metadata_buffer(self.metadata_buf)
    populator.load_associated_files([self.label_file_path])
    populator.populate()
