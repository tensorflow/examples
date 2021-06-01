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
"""APIs to train a model that can answer questions based on a predefined text."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile

import tensorflow as tf

from tensorflow_examples.lite.model_maker.core import compat
from tensorflow_examples.lite.model_maker.core.api import mm_export
from tensorflow_examples.lite.model_maker.core.export_format import ExportFormat
from tensorflow_examples.lite.model_maker.core.task import custom_model
from tensorflow_examples.lite.model_maker.core.task import model_spec as ms
from tensorflow_examples.lite.model_maker.core.task import model_util
from tensorflow_examples.lite.model_maker.core.task.metadata_writers.bert.question_answerer import metadata_writer_for_bert_question_answerer as metadata_writer


def _get_model_info(model_spec, vocab_file):
  """Gets the specific info for the question answer model."""
  return metadata_writer.QuestionAnswererInfo(
      name=model_spec.name + ' Question and Answerer',
      version='v1',
      description=metadata_writer.DEFAULT_DESCRIPTION,
      input_names=metadata_writer.bert_qa_inputs(
          ids_name=model_spec.tflite_input_name['ids'],
          mask_name=model_spec.tflite_input_name['mask'],
          segment_ids_name=model_spec.tflite_input_name['segment_ids']),
      output_names=metadata_writer.bert_qa_outputs(
          start_logits_name=model_spec.tflite_output_name['start_logits'],
          end_logits_name=model_spec.tflite_output_name['end_logits']),
      tokenizer_type=metadata_writer.Tokenizer.BERT_TOKENIZER,
      vocab_file=vocab_file)


@mm_export('question_answer.QuestionAnswer')
class QuestionAnswer(custom_model.CustomModel):
  """QuestionAnswer class for inference and exporting to tflite."""

  DEFAULT_EXPORT_FORMAT = (ExportFormat.TFLITE, ExportFormat.VOCAB)
  ALLOWED_EXPORT_FORMAT = (ExportFormat.TFLITE, ExportFormat.VOCAB,
                           ExportFormat.SAVED_MODEL)

  def train(self,
            train_data,
            epochs=None,
            batch_size=None,
            steps_per_epoch=None):
    """Feeds the training data for training."""
    if batch_size is None:
      batch_size = self.model_spec.default_batch_size

    if len(train_data) < batch_size:
      raise ValueError('The size of the train_data (%d) couldn\'t be smaller '
                       'than batch_size (%d). To solve this problem, set '
                       'the batch_size smaller or increase the size of the '
                       'train_data.' % (len(train_data), batch_size))

    train_ds = train_data.gen_dataset(batch_size, is_training=True)
    steps_per_epoch = model_util.get_steps_per_epoch(steps_per_epoch,
                                                     batch_size, train_data)
    if steps_per_epoch is not None:
      train_ds = train_ds.take(steps_per_epoch)
    self.model = self.model_spec.train(
        train_ds=train_ds, epochs=epochs, steps_per_epoch=steps_per_epoch)

    return self.model

  def create_model(self):
    self.model = self.model_spec.create_model()

  def evaluate(self,
               data,
               max_answer_length=30,
               null_score_diff_threshold=0.0,
               verbose_logging=False,
               output_dir=None):
    """Evaluate the model.

    Args:
      data: Data to be evaluated.
      max_answer_length: The maximum length of an answer that can be generated.
        This is needed because the start and end predictions are not conditioned
        on one another.
      null_score_diff_threshold: If null_score - best_non_null is greater than
        the threshold, predict null. This is only used for SQuAD v2.
      verbose_logging: If true, all of the warnings related to data processing
        will be printed. A number of warnings are expected for a normal SQuAD
        evaluation.
      output_dir: The output directory to save output to json files:
        predictions.json, nbest_predictions.json, null_odds.json. If None, skip
        saving to json files.

    Returns:
      A dict contains two metrics: Exact match rate and F1 score.
    """
    predict_batch_size = self.model_spec.predict_batch_size
    ds = data.gen_dataset(predict_batch_size, is_training=False)
    num_steps = int(len(data) / predict_batch_size)
    return self.model_spec.evaluate(
        self.model, None, ds, num_steps, data.examples, data.features,
        data.squad_file, data.version_2_with_negative, max_answer_length,
        null_score_diff_threshold, verbose_logging, output_dir)

  def evaluate_tflite(self,
                      tflite_filepath,
                      data,
                      max_answer_length=30,
                      null_score_diff_threshold=0.0,
                      verbose_logging=False,
                      output_dir=None):
    """Evaluate the model.

    Args:
      tflite_filepath: File path to the TFLite model.
      data: Data to be evaluated.
      max_answer_length: The maximum length of an answer that can be generated.
        This is needed because the start and end predictions are not conditioned
        on one another.
      null_score_diff_threshold: If null_score - best_non_null is greater than
        the threshold, predict null. This is only used for SQuAD v2.
      verbose_logging: If true, all of the warnings related to data processing
        will be printed. A number of warnings are expected for a normal SQuAD
        evaluation.
      output_dir: The output directory to save output to json files:
        predictions.json, nbest_predictions.json, null_odds.json. If None, skip
        saving to json files.

    Returns:
      A dict contains two metrics: Exact match rate and F1 score.
    """
    ds = data.gen_dataset(batch_size=1, is_training=False)
    return self.model_spec.evaluate(
        None, tflite_filepath, ds, len(data), data.examples, data.features,
        data.squad_file, data.version_2_with_negative, max_answer_length,
        null_score_diff_threshold, verbose_logging, output_dir)

  def _export_tflite(self,
                     tflite_filepath,
                     quantization_config='default',
                     with_metadata=True,
                     export_metadata_json_file=False):
    """Converts the retrained model to tflite format and saves it.

    Args:
      tflite_filepath: File path to save tflite model.
      quantization_config: Configuration for post-training quantization. If
        'default', sets the `quantization_config` by default according to
        `self.model_spec`. If None, exports the float tflite model without
        quantization.
      with_metadata: Whether the output tflite model contains metadata.
      export_metadata_json_file: Whether to export metadata in json file. If
        True, export the metadata in the same directory as tflite model.Used
        only if `with_metadata` is True.
    """
    if quantization_config == 'default':
      quantization_config = self.model_spec.get_default_quantization_config()

    # Sets batch size from None to 1 when converting to tflite.
    model_util.set_batch_size(self.model, batch_size=1)
    model_util.export_tflite(self.model, tflite_filepath, quantization_config,
                             self.model_spec.convert_from_saved_model_tf2)
    # Sets batch size back to None to support retraining later.
    model_util.set_batch_size(self.model, batch_size=None)

    if with_metadata:
      with tempfile.TemporaryDirectory() as temp_dir:
        tf.compat.v1.logging.info(
            'Vocab file is inside the TFLite model with metadata.')
        vocab_filepath = os.path.join(temp_dir, 'vocab.txt')
        self.model_spec.save_vocab(vocab_filepath)
        model_info = _get_model_info(self.model_spec, vocab_filepath)
        export_dir = os.path.dirname(tflite_filepath)
        populator = metadata_writer.MetadataPopulatorForBertQuestionAndAnswer(
            tflite_filepath, export_dir, model_info)
        populator.populate(export_metadata_json_file)

  @classmethod
  def create(cls,
             train_data,
             model_spec,
             batch_size=None,
             epochs=2,
             steps_per_epoch=None,
             shuffle=False,
             do_train=True):
    """Loads data and train the model for question answer.

    Args:
      train_data: Training data.
      model_spec: Specification for the model.
      batch_size: Batch size for training.
      epochs: Number of epochs for training.
      steps_per_epoch: Integer or None. Total number of steps (batches of
        samples) before declaring one epoch finished and starting the next
        epoch. If `steps_per_epoch` is None, the epoch will run until the input
        dataset is exhausted.
      shuffle: Whether the data should be shuffled.
      do_train: Whether to run training.

    Returns:
      An instance based on QuestionAnswer.
    """
    model_spec = ms.get(model_spec)
    if compat.get_tf_behavior() not in model_spec.compat_tf_versions:
      raise ValueError('Incompatible versions. Expect {}, but got {}.'.format(
          model_spec.compat_tf_versions, compat.get_tf_behavior()))

    model = cls(model_spec, shuffle=shuffle)

    if do_train:
      tf.compat.v1.logging.info('Retraining the models...')
      model.train(train_data, epochs, batch_size, steps_per_epoch)
    else:
      model.create_model()

    return model


# Shortcut function.
create = QuestionAnswer.create
mm_export('question_answer.create').export_constant(__name__, 'create')
