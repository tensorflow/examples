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
"""Helper methods to writes metadata into Bert models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import enum
import os

from tensorflow_examples.lite.model_maker.core.task.metadata_writers import metadata_writer

import flatbuffers
from tflite_support import metadata_schema_py_generated as _metadata_fb
from tflite_support import metadata as _metadata


class Tokenizer(enum.Enum):
  BERT_TOKENIZER = "BERT_TOKENIZER"
  SENTENCE_PIECE = "SENTENCE_PIECE"


def bert_qa_inputs(ids_name, mask_name, segment_ids_name):
  """Creates the input tensor names of a Bert model in order.

  The names correspond to `Tensor.name` in the TFLite schema. It helps to
  determine the tensor order when populating the metadata.

  Args:
    ids_name: name of the ids tensor, which represents the tokenized ids of
      input text as concatenated query and passage.
    mask_name: name of the mask tensor, which represents the mask with 1 for
      real tokens and 0 for padding tokens.
    segment_ids_name: name of the segment ids tensor, where 0 is for query and 1
      is for passage tokens.

  Returns:
    The input name list.
  """
  return [ids_name, mask_name, segment_ids_name]


class ModelSpecificInfo(object):
  """Holds information that is specificly tied to a Bert model."""

  def __init__(self,
               name,
               version,
               description,
               input_names,
               tokenizer_type,
               vocab_file=None,
               sp_model=None):
    """Constructor for ModelSpecificInfo.

    Args:
      name: name of the model in string.
      version: version of the model in string.
      description: description of the model.
      input_names: the name list returned by bert_qa_inputs.
      tokenizer_type: one of the tokenizer types in Tokenizer.
      vocab_file: the vocab file name to be packed into the model. If the
        tokenizer is BERT_TOKENIZER, the vocab file is required; if the
        tokenizer is SENTENCE_PIECE, the vocab file is optional.
      sp_model: the SentencePiece model file, only valid for the SENTENCE_PIECE
        tokenizer.
    """
    if tokenizer_type is Tokenizer.BERT_TOKENIZER:
      if vocab_file is None:
        raise ValueError(
            "The vocab file cannot be None for the BERT_TOKENIZER.")
    elif tokenizer_type is Tokenizer.SENTENCE_PIECE:
      if sp_model is None:
        raise ValueError(
            "The sentence piece model file cannot be None for the "
            "SENTENCE_PIECE tokenizer. The vocab file is optional though.")
    else:
      raise ValueError(
          "The tokenizer type, {0}, is unsupported.".format(tokenizer_type))

    self.name = name
    self.version = version
    self.description = description
    self.input_names = input_names
    self.tokenizer_type = tokenizer_type
    self.vocab_file = vocab_file
    self.sp_model = sp_model


class MetadataPopulatorForBert(metadata_writer.MetadataWriter):
  """Populates the metadata for a Bert model."""

  def __init__(self, model_file, export_directory, model_info):
    self.model_info = model_info
    model_dir_name = os.path.dirname(model_file)
    file_paths = []
    if model_info.vocab_file is not None:
      file_paths.append(os.path.join(model_dir_name, model_info.vocab_file))
    if model_info.sp_model is not None:
      file_paths.append(os.path.join(model_dir_name, model_info.sp_model))
    super(MetadataPopulatorForBert, self).__init__(model_file, export_directory,
                                                   file_paths)

  def _create_metadata(self):
    """Creates model metadata for bert models."""

    model_meta = _metadata_fb.ModelMetadataT()
    model_meta.name = self.model_info.name
    model_meta.description = self.model_info.description
    model_meta.version = self.model_info.version
    model_meta.author = "TensorFlow Lite Model Maker"
    model_meta.license = ("Apache License. Version 2.0 "
                          "http://www.apache.org/licenses/LICENSE-2.0.")

    # Creates the tokenizer info.
    if self.model_info.tokenizer_type is Tokenizer.BERT_TOKENIZER:
      tokenizer = self._create_bert_tokenizer()
    elif self.model_info.tokenizer_type is Tokenizer.SENTENCE_PIECE:
      tokenizer = self._create_sentence_piece_tokenizer()
    else:
      raise ValueError("The tokenizer type, {0}, is unsupported.".format(
          self.model_info.tokenizer_type))

    # Creates subgraph info.
    subgraph = _metadata_fb.SubGraphMetadataT()
    subgraph.inputTensorMetadata = self._create_input_metadata()
    subgraph.outputTensorMetadata = self._create_output_metadata()
    subgraph.inputProcessUnits = [tokenizer]
    model_meta.subgraphMetadata = [subgraph]

    b = flatbuffers.Builder(0)
    b.Finish(
        model_meta.Pack(b),
        _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER)
    self.metadata_buf = b.Output()

  def _create_input_metadata(self):
    """Creates the input metadata for a Bert model.

    Returns:
      A list of the three input tensor metadata in flatbuffer objects.
    """

    # Creates inputs info.
    ids_meta = _metadata_fb.TensorMetadataT()
    ids_meta.name = "ids"
    ids_meta.description = (
        "Tokenized ids of input text.")
    ids_meta.content = _metadata_fb.ContentT()
    ids_meta.content.contentPropertiesType = (
        _metadata_fb.ContentProperties.FeatureProperties)
    ids_meta.content.contentProperties = _metadata_fb.FeaturePropertiesT()

    mask_meta = _metadata_fb.TensorMetadataT()
    mask_meta.name = "mask"
    mask_meta.description = ("Mask with 1 for real tokens and 0 for padding "
                             "tokens.")
    mask_meta.content = _metadata_fb.ContentT()
    mask_meta.content.contentPropertiesType = (
        _metadata_fb.ContentProperties.FeatureProperties)
    mask_meta.content.contentProperties = _metadata_fb.FeaturePropertiesT()

    segment_meta = _metadata_fb.TensorMetadataT()
    segment_meta.name = "segment_ids"
    segment_meta.description = (
        "0 for the first sequence, 1 for the second sequence if exists.")
    segment_meta.content = _metadata_fb.ContentT()
    segment_meta.content.contentPropertiesType = (
        _metadata_fb.ContentProperties.FeatureProperties)
    segment_meta.content.contentProperties = _metadata_fb.FeaturePropertiesT()

    # The order of input_metadata should match the order of tensor names in
    # InputTensorNames.
    input_metadata = [ids_meta, mask_meta, segment_meta]
    # Order the tensor metadata according to the input tensor order.
    ordered_input_names = self._get_input_tensor_names()
    return self._order_tensor_metadata_with_names(input_metadata,
                                                  self.model_info.input_names,
                                                  ordered_input_names)

  @abc.abstractmethod
  def _create_output_metadata(self):
    """Creates the output metadata for a Bert model.

    Returns:
      A list of the output tensor metadata in flatbuffer objects.
    """
    pass

  def _create_bert_tokenizer(self):
    vocab = _metadata_fb.AssociatedFileT()
    vocab.name = os.path.basename(self.model_info.vocab_file)
    vocab.description = "Vocabulary file for the BertTokenizer."
    vocab.type = _metadata_fb.AssociatedFileType.VOCABULARY
    tokenizer = _metadata_fb.ProcessUnitT()
    tokenizer.optionsType = _metadata_fb.ProcessUnitOptions.BertTokenizerOptions
    tokenizer.options = _metadata_fb.BertTokenizerOptionsT()
    tokenizer.options.vocabFile = [vocab]
    return tokenizer

  def _create_sentence_piece_tokenizer(self):
    sp_model = _metadata_fb.AssociatedFileT()
    sp_model.name = os.path.basename(self.model_info.sp_model)
    sp_model.description = "The sentence piece model file."
    if self.model_info.vocab_file is not None:
      vocab = _metadata_fb.AssociatedFileT()
      vocab.name = os.path.basename(self.model_info.vocab_file)
      vocab.description = (
          "Vocabulary file for the SentencePiece tokenizer. This file is "
          "optional during tokenization, while the sentence piece model is "
          "mandatory.")
      vocab.type = _metadata_fb.AssociatedFileType.VOCABULARY
    tokenizer = _metadata_fb.ProcessUnitT()
    tokenizer.optionsType = (
        _metadata_fb.ProcessUnitOptions.SentencePieceTokenizerOptions)
    tokenizer.options = _metadata_fb.SentencePieceTokenizerOptionsT()
    tokenizer.options.sentencePieceModel = [sp_model]
    if vocab:
      tokenizer.options.vocabFile = [vocab]
    return tokenizer
