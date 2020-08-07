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
"""QuestionAnswer class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_examples.lite.model_maker.core import compat
from tensorflow_examples.lite.model_maker.core.export_format import ExportFormat
from tensorflow_examples.lite.model_maker.core.task import custom_model
from tensorflow_examples.lite.model_maker.core.task import model_util


def create(train_data,
           model_spec,
           batch_size=None,
           epochs=2,
           shuffle=False,
           do_train=True):
  """Loads data and train the model for question answer.

  Args:
    train_data: Training data.
    model_spec: Specification for the model.
    batch_size: Batch size for training.
    epochs: Number of epochs for training.
    shuffle: Whether the data should be shuffled.
    do_train: Whether to run training.

  Returns:
    object of QuestionAnswer class.
  """
  if compat.get_tf_behavior() not in model_spec.compat_tf_versions:
    raise ValueError('Incompatible versions. Expect {}, but got {}.'.format(
        model_spec.compat_tf_versions, compat.get_tf_behavior()))

  model = QuestionAnswer(model_spec, shuffle=shuffle)

  if do_train:
    tf.compat.v1.logging.info('Retraining the models...')
    model.train(train_data, epochs, batch_size)
  else:
    model.create_model()

  return model


class QuestionAnswer(custom_model.CustomModel):
  """QuestionAnswer class for inference and exporting to tflite."""

  DEFAULT_EXPORT_FORMAT = [ExportFormat.TFLITE, ExportFormat.VOCAB]
  ALLOWED_EXPORT_FORMAT = [
      ExportFormat.TFLITE, ExportFormat.VOCAB, ExportFormat.SAVED_MODEL
  ]

  def preprocess(self, raw_text, label):
    """Preprocess the text."""
    # TODO(yuqili): remove this method once preprocess for image classifier is
    # also moved to DataLoader part.
    return raw_text, label

  def train(self, train_data, epochs=None, batch_size=None):
    """Feeds the training data for training."""
    if batch_size is None:
      batch_size = self.model_spec.default_batch_size

    if train_data.size < batch_size:
      raise ValueError('The size of the train_data (%d) couldn\'t be smaller '
                       'than batch_size (%d). To solve this problem, set '
                       'the batch_size smaller or increase the size of the '
                       'train_data.' % (train_data.size, batch_size))

    train_input_fn, steps_per_epoch = self._get_input_fn_and_steps(
        train_data, batch_size, is_training=True)

    self.model = self.model_spec.train(train_input_fn, epochs, steps_per_epoch)

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
    input_fn = self._get_dataset_fn(data, predict_batch_size, is_training=False)
    num_steps = int(data.size / predict_batch_size)
    return self.model_spec.evaluate(
        self.model, None, input_fn, num_steps, data.examples, data.features,
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
    input_fn = self._get_dataset_fn(
        data, global_batch_size=1, is_training=False)
    return self.model_spec.evaluate(
        None, tflite_filepath, input_fn, data.size, data.examples,
        data.features, data.squad_file, data.version_2_with_negative,
        max_answer_length, null_score_diff_threshold, verbose_logging,
        output_dir)

  def export(self,
             export_dir,
             tflite_filename='model.tflite',
             vocab_filename='vocab',
             saved_model_filename='saved_model',
             export_format=None,
             **kwargs):
    """Converts the retrained model based on `export_format`.

    Args:
      export_dir: The directory to save exported files.
      tflite_filename: File name to save tflite model. The full export path is
        {export_dir}/{tflite_filename}.
      vocab_filename: File name to save vocabulary.  The full export path is
        {export_dir}/{vocab_filename}.
      saved_model_filename: Path to SavedModel or H5 file to save the model. The
        full export path is
        {export_dir}/{saved_model_filename}/{saved_model.pb|assets|variables}.
      export_format: List of export format that could be saved_model, tflite,
        label, vocab.
      **kwargs: Other parameters like `quantized` for TFLITE model.
    """
    super(QuestionAnswer, self).export(
        export_dir,
        tflite_filename=tflite_filename,
        vocab_filename=vocab_filename,
        saved_model_filename=saved_model_filename,
        export_format=export_format,
        **kwargs)

  def _export_tflite(self, tflite_filepath, quantization_config=None):
    """Converts the retrained model to tflite format and saves it.

    Args:
      tflite_filepath: File path to save tflite model.
      quantization_config: Configuration for post-training quantization.
    """
    # Sets batch size from None to 1 when converting to tflite.
    model_util.set_batch_size(self.model, batch_size=1)
    model_util.export_tflite(self.model, tflite_filepath, quantization_config,
                             self._gen_dataset,
                             self.model_spec.convert_from_saved_model_tf2)
    # Sets batch size back to None to support retraining later.
    model_util.set_batch_size(self.model, batch_size=None)
