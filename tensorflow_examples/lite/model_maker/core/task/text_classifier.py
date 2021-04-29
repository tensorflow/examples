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
"""APIs to train a text classification model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile

import tensorflow as tf

from tensorflow_examples.lite.model_maker.core import compat
from tensorflow_examples.lite.model_maker.core.api import mm_export
from tensorflow_examples.lite.model_maker.core.export_format import ExportFormat
from tensorflow_examples.lite.model_maker.core.task import classification_model
from tensorflow_examples.lite.model_maker.core.task import model_spec as ms
from tensorflow_examples.lite.model_maker.core.task import model_util
from tensorflow_examples.lite.model_maker.core.task.metadata_writers.bert.text_classifier import metadata_writer_for_bert_text_classifier as bert_metadata_writer
from tensorflow_examples.lite.model_maker.core.task.metadata_writers.text_classifier import metadata_writer_for_text_classifier as metadata_writer
from tensorflow_examples.lite.model_maker.core.task.model_spec import text_spec


@mm_export('text_classifier.create')
def create(train_data,
           model_spec='average_word_vec',
           validation_data=None,
           batch_size=None,
           epochs=3,
           shuffle=False,
           do_train=True):
  """Loads data and train the model for test classification.

  Args:
    train_data: Training data.
    model_spec: Specification for the model.
    validation_data: Validation data. If None, skips validation process.
    batch_size: Batch size for training.
    epochs: Number of epochs for training.
    shuffle: Whether the data should be shuffled.
    do_train: Whether to run training.

  Returns:
    TextClassifier
  """
  model_spec = ms.get(model_spec)
  if compat.get_tf_behavior() not in model_spec.compat_tf_versions:
    raise ValueError('Incompatible versions. Expect {}, but got {}.'.format(
        model_spec.compat_tf_versions, compat.get_tf_behavior()))

  text_classifier = TextClassifier(
      model_spec,
      train_data.index_to_label,
      shuffle=shuffle)

  if do_train:
    tf.compat.v1.logging.info('Retraining the models...')
    text_classifier.train(train_data, validation_data, epochs, batch_size)
  else:
    text_classifier.create_model()

  return text_classifier


def _get_bert_model_info(model_spec, vocab_file, label_file):
  return bert_metadata_writer.ClassifierSpecificInfo(
      name=model_spec.name + ' text classifier',
      version='v1',
      description=bert_metadata_writer.DEFAULT_DESCRIPTION,
      input_names=bert_metadata_writer.bert_qa_inputs(
          ids_name=model_spec.tflite_input_name['ids'],
          mask_name=model_spec.tflite_input_name['mask'],
          segment_ids_name=model_spec.tflite_input_name['segment_ids']),
      tokenizer_type=bert_metadata_writer.Tokenizer.BERT_TOKENIZER,
      vocab_file=vocab_file,
      label_file=label_file)


def _get_model_info(model_name):
  return metadata_writer.ModelSpecificInfo(
      name=model_name + ' text classifier',
      description='Classify text into predefined categories.',
      version='v1')


@mm_export('text_classifier.TextClassifier')
class TextClassifier(classification_model.ClassificationModel):
  """TextClassifier class for inference and exporting to tflite."""

  DEFAULT_EXPORT_FORMAT = (ExportFormat.TFLITE, ExportFormat.LABEL,
                           ExportFormat.VOCAB)
  ALLOWED_EXPORT_FORMAT = (ExportFormat.TFLITE, ExportFormat.LABEL,
                           ExportFormat.VOCAB, ExportFormat.SAVED_MODEL,
                           ExportFormat.TFJS)

  def __init__(self,
               model_spec,
               index_to_label,
               shuffle=True):
    """Init function for TextClassifier class.

    Args:
      model_spec: Specification for the model.
      index_to_label: A list that map from index to label class name.
      shuffle: Whether the data should be shuffled.
    """
    super(TextClassifier, self).__init__(
        model_spec,
        index_to_label,
        shuffle,
        train_whole_model=True)

  def create_model(self):
    self.model = self.model_spec.create_model(self.num_classes)

  def train(self,
            train_data,
            validation_data=None,
            epochs=None,
            batch_size=None):
    """Feeds the training data for training."""
    if batch_size is None:
      batch_size = self.model_spec.default_batch_size

    if len(train_data) < batch_size:
      raise ValueError('The size of the train_data (%d) couldn\'t be smaller '
                       'than batch_size (%d). To solve this problem, set '
                       'the batch_size smaller or increase the size of the '
                       'train_data.' % (len(train_data), batch_size))

    train_input_fn, steps_per_epoch = self._get_input_fn_and_steps(
        train_data, batch_size, is_training=True)
    validation_input_fn, validation_steps = self._get_input_fn_and_steps(
        validation_data, batch_size, is_training=False)

    self.model = self.model_spec.run_classifier(
        train_input_fn,
        validation_input_fn,
        epochs,
        steps_per_epoch,
        validation_steps,
        self.num_classes,
        callbacks=self._keras_callbacks(model_dir=self.model_spec.model_dir))

    return self.model

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
        tf.compat.v1.logging.info('Vocab file and label file are inside the '
                                  'TFLite model with metadata.')
        vocab_filepath = os.path.join(temp_dir, 'vocab.txt')
        self.model_spec.save_vocab(vocab_filepath)
        label_filepath = os.path.join(temp_dir, 'labels.txt')
        self._export_labels(label_filepath)

        export_dir = os.path.dirname(tflite_filepath)
        if isinstance(self.model_spec, text_spec.BertClassifierModelSpec):
          model_info = _get_bert_model_info(self.model_spec, vocab_filepath,
                                            label_filepath)
          populator = bert_metadata_writer.MetadataPopulatorForBertTextClassifier(
              tflite_filepath, export_dir, model_info)
        elif isinstance(self.model_spec, text_spec.AverageWordVecModelSpec):
          model_info = _get_model_info(self.model_spec.name)
          populator = metadata_writer.MetadataPopulatorForTextClassifier(
              tflite_filepath, export_dir, model_info, label_filepath,
              vocab_filepath)
        else:
          raise ValueError('Model Specification is not supported to writing '
                           'metadata into TFLite. Please set '
                           '`with_metadata=False` or write metadata by '
                           'yourself.')
        populator.populate(export_metadata_json_file)
