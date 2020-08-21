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
"""Writes metadata and tokenize file to the Bert QA models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_examples.lite.model_maker.core.task.metadata_writers.bert.metadata_writer_for_bert import bert_qa_inputs
from tensorflow_examples.lite.model_maker.core.task.metadata_writers.bert.metadata_writer_for_bert import MetadataPopulatorForBert
from tensorflow_examples.lite.model_maker.core.task.metadata_writers.bert.metadata_writer_for_bert import ModelSpecificInfo
from tensorflow_examples.lite.model_maker.core.task.metadata_writers.bert.metadata_writer_for_bert import Tokenizer
from tflite_support import metadata_schema_py_generated as _metadata_fb


def bert_qa_outputs(start_logits_name, end_logits_name):
  """Creates the output tensor names of a Bert question answerer model in order.

  The names correspond to `Tensor.name` in the TFLite schema. It helps to
  determine the tensor order when populating the metadata.

  Args:
    start_logits_name: name of the start logits tensor, which represents the
      start position of the answer span.
    end_logits_name: name of the end logits tensor, which represents the end
      position of the answer span.

  Returns:
    The output name list.
  """
  return [start_logits_name, end_logits_name]


class QuestionAnswererInfo(ModelSpecificInfo):
  """Holds information specificly tied to a Bert question answerer model."""

  def __init__(self,
               name,
               version,
               description,
               input_names,
               output_names,
               tokenizer_type,
               vocab_file=None,
               sp_model=None):
    """Constructor for ModelSpecificInfo.

    Args:
      name: name of the model in string.
      version: version of the model in string.
      description: description of the model.
      input_names: the name list returned by bert_qa_inputs.
      output_names: the name list returned by bert_qa_outputs.
      tokenizer_type: one of the tokenizer types in Tokenizer.
      vocab_file: the vocab file name to be packed into the model. If the
        tokenizer is BERT_TOKENIZER, the vocab file is required; if the
        tokenizer is SENTENCE_PIECE, the vocab file is optional.
      sp_model: the SentencePiece model file, only valid for the SENTENCE_PIECE
        tokenizer.
    """
    self.output_names = output_names
    super(QuestionAnswererInfo,
          self).__init__(name, version, description, input_names,
                         tokenizer_type, vocab_file, sp_model)


DEFAULT_DESCRIPTION = (
    "Answers questions based on the content of a given "
    "passage. To integrate the model into your app, try the "
    "`BertQuestionAnswerer` API in the TensorFlow Lite Task "
    "library. `BertQuestionAnswerer` takes a passage string and a "
    "query string, and returns the answer strings. It encapsulates "
    "the processing logic of inputs and outputs and runs the "
    "inference with the best practice.")

DEFAULT_INPUT_NAMES = bert_qa_inputs(
    ids_name="input_ids",
    mask_name="input_mask",
    segment_ids_name="segment_ids")

DEFAULT_OUTPUT_NAMES = bert_qa_outputs(
    start_logits_name="start_logits", end_logits_name="end_logits")

_MODEL_INFO = {
    "mobilebert_float.tflite":
        QuestionAnswererInfo(
            name="MobileBert Question and Answerer",
            version="v1",
            description=DEFAULT_DESCRIPTION,
            input_names=DEFAULT_INPUT_NAMES,
            output_names=DEFAULT_OUTPUT_NAMES,
            tokenizer_type=Tokenizer.BERT_TOKENIZER,
            vocab_file="vocab.txt"),
    "albert_float.tflite":
        QuestionAnswererInfo(
            name="Albert Question and Answerer",
            version="v1",
            description=DEFAULT_DESCRIPTION,
            input_names=DEFAULT_INPUT_NAMES,
            output_names=DEFAULT_OUTPUT_NAMES,
            tokenizer_type=Tokenizer.SENTENCE_PIECE,
            vocab_file="30k-clean.vocab",
            sp_model="30k-clean.model"),
}


class MetadataPopulatorForBertQuestionAndAnswer(MetadataPopulatorForBert):
  """Populates the metadata for a Bert QA model."""

  def _create_output_metadata(self):
    """Creates the output metadata for a Bert QA model."""

    # Creates outputs info.
    end_meta = _metadata_fb.TensorMetadataT()
    end_meta.name = "end_logits"
    end_meta.description = (
        "logits over the sequence which indicates the"
        " end position of the answer span with closed interval.")
    end_meta.content = _metadata_fb.ContentT()
    end_meta.content.contentPropertiesType = (
        _metadata_fb.ContentProperties.FeatureProperties)
    end_meta.content.contentProperties = _metadata_fb.FeaturePropertiesT()

    start_meta = _metadata_fb.TensorMetadataT()
    start_meta.name = "start_logits"
    start_meta.description = (
        "logits over the sequence which indicates "
        "the start position of the answer span with closed interval.")
    start_meta.content = _metadata_fb.ContentT()
    start_meta.content.contentPropertiesType = (
        _metadata_fb.ContentProperties.FeatureProperties)
    start_meta.content.contentProperties = _metadata_fb.FeaturePropertiesT()

    # The order of output_metadata should match the order of tensor names in
    # OutputTensorNames.
    output_metadata = [start_meta, end_meta]
    # Order the tensor metadata according to the output tensor order.
    ordered_output_names = self._get_output_tensor_names()
    return self._order_tensor_metadata_with_names(output_metadata,
                                                  self.model_info.output_names,
                                                  ordered_output_names)
