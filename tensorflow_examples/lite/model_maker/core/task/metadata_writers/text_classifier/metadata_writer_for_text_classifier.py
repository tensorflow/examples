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
"""Writes metadata and label file to the text classifier models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow_examples.lite.model_maker.core.task.metadata_writers import metadata_writer
import flatbuffers
# pylint: disable=g-direct-tensorflow-import
from tflite_support import metadata as _metadata
from tflite_support import metadata_schema_py_generated as _metadata_fb
# pylint: enable=g-direct-tensorflow-import


# Default settings that must align with Model Maker.
# https://www.tensorflow.org/lite/tutorials/model_maker_text_classification
# Name for associated files.
DEFAULT_LABEL_FILE = "labels.txt"
DEFAULT_VOCAB_FILE = "vocab.txt"
# Split pattern used in the tokenizer.
DEFAULT_DELIM_REGEX_PATTERN = r"[^\w\']+"


class ModelSpecificInfo(object):
  """Holds information that is specificly tied to a text classifier."""

  def __init__(self,
               name,
               version,
               description,
               delim_regex_pattern=DEFAULT_DELIM_REGEX_PATTERN,
               label_file=DEFAULT_LABEL_FILE,
               vocab_file=DEFAULT_VOCAB_FILE):
    self.name = name
    self.version = version
    self.description = description
    self.delim_regex_pattern = delim_regex_pattern
    self.label_file = label_file
    self.vocab_file = vocab_file


_MODEL_INFO = {
    # AverageWordVector model trained on IMDB movie reviews dataset.
    "text_classification.tflite":
        ModelSpecificInfo(
            name="Sentiment Analyzer (AverageWordVecModelSpec)",
            description="Detect if the input text's sentiment is positive or "
            "negative. The model was trained on the IMDB Movie Reviews dataset "
            "so it is more accurate when input text is a movie review.",
            version="v1"),
}


class MetadataPopulatorForTextClassifier(metadata_writer.MetadataWriter):
  """Populates the metadata for a text classifier."""

  def __init__(self, model_file, export_directory, model_info, label_file_path,
               vocab_file_path):
    self.model_info = model_info
    super(MetadataPopulatorForTextClassifier,
          self).__init__(model_file, export_directory,
                         [label_file_path, vocab_file_path])

  def _create_metadata(self):
    """Creates the metadata for a text classifier."""

    # Creates model info.
    model_meta = _metadata_fb.ModelMetadataT()
    model_meta.name = self.model_info.name
    model_meta.description = self.model_info.description
    model_meta.version = self.model_info.version
    model_meta.author = "TensorFlow"
    model_meta.license = ("Apache License. Version 2.0 "
                          "http://www.apache.org/licenses/LICENSE-2.0.")

    # Creates input info.
    input_meta = _metadata_fb.TensorMetadataT()
    input_meta.name = "input_text"
    input_meta.description = (
        "Embedding vectors representing the input text to be classified. The "
        "input need to be converted from raw text to embedding vectors using "
        "the attached dictionary file.")
    # Create the vocab file.
    vocab_file = _metadata_fb.AssociatedFileT()
    vocab_file.name = os.path.basename(self.associated_files[1])
    vocab_file.description = ("Vocabulary file to convert natural language "
                              "words to embedding vectors.")
    vocab_file.type = _metadata_fb.AssociatedFileType.VOCABULARY

    # Create the RegexTokenizer.
    tokenizer = _metadata_fb.ProcessUnitT()
    tokenizer.optionsType = (
        _metadata_fb.ProcessUnitOptions.RegexTokenizerOptions)
    tokenizer.options = _metadata_fb.RegexTokenizerOptionsT()
    tokenizer.options.delimRegexPattern = self.model_info.delim_regex_pattern
    tokenizer.options.vocabFile = [vocab_file]

    input_meta.content = _metadata_fb.ContentT()
    input_meta.content.contentPropertiesType = (
        _metadata_fb.ContentProperties.FeatureProperties)
    input_meta.content.contentProperties = _metadata_fb.FeaturePropertiesT()
    input_meta.processUnits = [tokenizer]

    # Creates output info.
    output_meta = _metadata_fb.TensorMetadataT()
    output_meta.name = "probability"
    output_meta.description = "Probabilities of the labels respectively."
    output_meta.content = _metadata_fb.ContentT()
    output_meta.content.contentProperties = _metadata_fb.FeaturePropertiesT()
    output_meta.content.contentPropertiesType = (
        _metadata_fb.ContentProperties.FeatureProperties)
    output_stats = _metadata_fb.StatsT()
    output_stats.max = [1.0]
    output_stats.min = [0.0]
    output_meta.stats = output_stats
    label_file = _metadata_fb.AssociatedFileT()
    label_file.name = os.path.basename(self.associated_files[0])
    label_file.description = ("Labels for the categories that the model can "
                              "classify.")
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
