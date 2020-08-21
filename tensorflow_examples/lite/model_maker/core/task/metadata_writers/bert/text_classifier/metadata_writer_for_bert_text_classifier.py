# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Writes metadata and tokenize file to the Bert text classifier models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow_examples.lite.model_maker.core.task.metadata_writers.bert.metadata_writer_for_bert import bert_qa_inputs
from tensorflow_examples.lite.model_maker.core.task.metadata_writers.bert.metadata_writer_for_bert import MetadataPopulatorForBert
from tensorflow_examples.lite.model_maker.core.task.metadata_writers.bert.metadata_writer_for_bert import ModelSpecificInfo
from tensorflow_examples.lite.model_maker.core.task.metadata_writers.bert.metadata_writer_for_bert import Tokenizer
from tflite_support import metadata_schema_py_generated as _metadata_fb


class ClassifierSpecificInfo(ModelSpecificInfo):
  """Holds information specificly tied to a Bert text classifier model."""

  def __init__(self,
               name,
               version,
               description,
               input_names,
               tokenizer_type,
               label_file,
               vocab_file=None,
               sp_model=None):
    """Constructor for ModelSpecificInfo.

    Args:
      name: name of the model in string.
      version: version of the model in string.
      description: description of the model.
      input_names: an InputTensorNames object.
      tokenizer_type: one of the tokenizer types in Tokenizer.
      label_file: the label file of classification catergories.
      vocab_file: the vocab file name to be packed into the model. If the
        tokenizer is BERT_TOKENIZER, the vocab file is required; if the
        tokenizer is SENTENCE_PIECE, the vocab file is optional.
      sp_model: the SentencePiece model file, only valid for the SENTENCE_PIECE
        tokenizer.
    """
    self.label_file = label_file
    super(ClassifierSpecificInfo,
          self).__init__(name, version, description, input_names,
                         tokenizer_type, vocab_file, sp_model)


DEFAULT_DESCRIPTION = (
    "Classifies the input string based on the known catergories. To integrate "
    "the model into your app, try the `BertNLClassifier` API in the TensorFlow "
    "Lite Task library. `BertNLClassifier` takes an input string, and returns "
    "the classified label with probability. It encapsulates the processing "
    "logic of inputs and outputs and runs the inference with the best "
    "practice.")

_MODEL_INFO = {
    "sst2_mobilebert_quant.tflite":
        ClassifierSpecificInfo(
            name="MobileBert text classifier",
            version="v1",
            description=DEFAULT_DESCRIPTION,
            input_names=bert_qa_inputs(
                ids_name="serving_default_input_word_ids:0",
                mask_name="serving_default_input_mask:0",
                segment_ids_name="serving_default_input_type_ids:0"),
            tokenizer_type=Tokenizer.BERT_TOKENIZER,
            vocab_file="vocab.txt",
            label_file="labels.txt")
}


class MetadataPopulatorForBertTextClassifier(MetadataPopulatorForBert):
  """Populates the metadata for a Bert text classifier model."""

  def __init__(self, model_file, export_directory, model_info):
    self.model_info = model_info
    model_dir_name = os.path.dirname(model_file)
    file_paths = []
    if model_info.vocab_file is not None:
      file_paths.append(os.path.join(model_dir_name, model_info.vocab_file))
    if model_info.sp_model is not None:
      file_paths.append(os.path.join(model_dir_name, model_info.sp_model))
    file_paths.append(os.path.join(model_dir_name, model_info.label_file))
    self.model_file = model_file
    self.export_directory = export_directory
    self.associated_files = file_paths
    self.metadata_buf = None

  def _create_output_metadata(self):
    """Creates the output metadata for a Bert text classifier model."""

    # Create output info.
    output_meta = _metadata_fb.TensorMetadataT()
    output_meta.name = "probability"
    output_meta.description = "Probabilities of labels respectively."
    output_meta.content = _metadata_fb.ContentT()
    output_meta.content.contentProperties = _metadata_fb.FeaturePropertiesT()
    output_meta.content.contentPropertiesType = (
        _metadata_fb.ContentProperties.FeatureProperties)
    label_file = _metadata_fb.AssociatedFileT()
    label_file.name = os.path.basename(self.model_info.label_file)
    label_file.description = ("Labels for classification categories.")
    label_file.type = _metadata_fb.AssociatedFileType.TENSOR_AXIS_LABELS
    output_meta.associatedFiles = [label_file]
    return [output_meta]
