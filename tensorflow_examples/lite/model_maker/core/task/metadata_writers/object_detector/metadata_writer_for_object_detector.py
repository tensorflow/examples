# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Writes metadata and label file to the object detector models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow_examples.lite.model_maker.core.task.metadata_writers import metadata_writer
import flatbuffers
# pylint: disable=g-direct-tensorflow-import
from tflite_support import metadata_schema_py_generated as _metadata_fb
from tflite_support import metadata as _metadata
# pylint: enable=g-direct-tensorflow-import


class ModelSpecificInfo(object):
  """Holds information that is specificly tied to an object detector."""

  def __init__(self, name, version, image_width, image_height, image_min,
               image_max, mean, std):
    self.name = name
    self.version = version
    self.image_width = image_width
    self.image_height = image_height
    self.image_min = image_min
    self.image_max = image_max
    self.mean = mean
    self.std = std


class MetadataPopulatorForObjectDetector(metadata_writer.MetadataWriter):
  """Populates the metadata for an object detector."""

  def __init__(self, model_file, export_directory, model_info, label_file_path):
    self.model_info = model_info
    super(MetadataPopulatorForObjectDetector,
          self).__init__(model_file, export_directory, [label_file_path])

  # TODO(b/150647930): refine MetaData API to __init__ with properties.
  def _create_metadata(self):
    """Creates the metadata for an object detector."""

    model_meta = _metadata_fb.ModelMetadataT()
    model_meta.name = self.model_info.name
    model_meta.description = (
        "Identify which of a known set of objects might be present and provide "
        "information about their positions within the given image or a video "
        "stream.")
    model_meta.version = self.model_info.version
    model_meta.author = "TensorFlow Lite Model Maker"
    model_meta.license = ("Apache License. Version 2.0 "
                          "http://www.apache.org/licenses/LICENSE-2.0.")

    # Creates input info.
    input_meta = _metadata_fb.TensorMetadataT()
    input_meta.name = "image"
    input_meta.description = (
        "Input image to be detected. The expected image is {0} x {1}, with "
        "three channels (red, blue, and green) per pixel. Each value in the "
        "tensor is a single byte between 0 and 255.".format(
            self.model_info.image_width, self.model_info.image_height))
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

    # Creates outputs info.
    output_location_meta = _metadata_fb.TensorMetadataT()
    output_location_meta.name = "location"
    output_location_meta.description = "The locations of the detected boxes."
    output_location_meta.content = _metadata_fb.ContentT()
    output_location_meta.content.contentPropertiesType = (
        _metadata_fb.ContentProperties.BoundingBoxProperties)
    output_location_meta.content.contentProperties = (
        _metadata_fb.BoundingBoxPropertiesT())
    output_location_meta.content.contentProperties.index = [1, 0, 3, 2]
    output_location_meta.content.contentProperties.type = (
        _metadata_fb.BoundingBoxType.BOUNDARIES)
    output_location_meta.content.contentProperties.coordinateType = (
        _metadata_fb.CoordinateType.RATIO)
    output_location_meta.content.range = _metadata_fb.ValueRangeT()
    output_location_meta.content.range.min = 2
    output_location_meta.content.range.max = 2

    output_class_meta = _metadata_fb.TensorMetadataT()
    output_class_meta.name = "category"
    output_class_meta.description = "The categories of the detected boxes."
    output_class_meta.content = _metadata_fb.ContentT()
    output_class_meta.content.contentPropertiesType = (
        _metadata_fb.ContentProperties.FeatureProperties)
    output_class_meta.content.contentProperties = (
        _metadata_fb.FeaturePropertiesT())
    output_class_meta.content.range = _metadata_fb.ValueRangeT()
    output_class_meta.content.range.min = 2
    output_class_meta.content.range.max = 2
    label_file = _metadata_fb.AssociatedFileT()
    label_file.name = os.path.basename(self.associated_files[0])
    label_file.description = "Label of objects that this model can recognize."
    label_file.type = _metadata_fb.AssociatedFileType.TENSOR_VALUE_LABELS
    output_class_meta.associatedFiles = [label_file]

    output_score_meta = _metadata_fb.TensorMetadataT()
    output_score_meta.name = "score"
    output_score_meta.description = "The scores of the detected boxes."
    output_score_meta.content = _metadata_fb.ContentT()
    output_score_meta.content.contentPropertiesType = (
        _metadata_fb.ContentProperties.FeatureProperties)
    output_score_meta.content.contentProperties = (
        _metadata_fb.FeaturePropertiesT())
    output_score_meta.content.range = _metadata_fb.ValueRangeT()
    output_score_meta.content.range.min = 2
    output_score_meta.content.range.max = 2

    output_number_meta = _metadata_fb.TensorMetadataT()
    output_number_meta.name = "number of detections"
    output_number_meta.description = "The number of the detected boxes."
    output_number_meta.content = _metadata_fb.ContentT()
    output_number_meta.content.contentPropertiesType = (
        _metadata_fb.ContentProperties.FeatureProperties)
    output_number_meta.content.contentProperties = (
        _metadata_fb.FeaturePropertiesT())

    # Creates subgraph info.
    group = _metadata_fb.TensorGroupT()
    group.name = "detection result"
    group.tensorNames = [
        output_location_meta.name, output_class_meta.name,
        output_score_meta.name
    ]
    subgraph = _metadata_fb.SubGraphMetadataT()
    subgraph.inputTensorMetadata = [input_meta]
    subgraph.outputTensorMetadata = [
        output_location_meta, output_class_meta, output_score_meta,
        output_number_meta
    ]
    subgraph.outputTensorGroups = [group]
    model_meta.subgraphMetadata = [subgraph]

    b = flatbuffers.Builder(0)
    b.Finish(
        model_meta.Pack(b),
        _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER)
    self.metadata_buf = b.Output()
