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
"""TextClassier class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf

from tensorflow_examples.lite.model_maker.core import compat
from tensorflow_examples.lite.model_maker.core import model_export_format as mef
from tensorflow_examples.lite.model_maker.core.task import classification_model
from tensorflow_examples.lite.model_maker.core.task import model_spec as ms


def create(train_data,
           model_export_format=mef.ModelExportFormat.TFLITE,
           model_spec=ms.AverageWordVecModelSpec(),
           shuffle=False,
           batch_size=32,
           epochs=None,
           validation_data=None):
  """Loads data and train the model for test classification.

  Args:
    train_data: Training data.
    model_export_format: Model export format such as saved_model / tflite.
    model_spec: Specification for the model.
    shuffle: Whether the data should be shuffled.
    batch_size: Batch size for training.
    epochs: Number of epochs for training.
    validation_data: Validation data. If None, skips validation process.

  Returns:
    TextClassifier
  """
  if compat.get_tf_behavior() not in model_spec.compat_tf_versions:
    raise ValueError('Incompatible versions. Expect {}, but got {}.'.format(
        model_spec.compat_tf_versions, compat.get_tf_behavior()))

  text_classifier = TextClassifier(
      model_export_format,
      model_spec,
      train_data.index_to_label,
      train_data.num_classes,
      shuffle=shuffle)

  tf.compat.v1.logging.info('Retraining the models...')
  text_classifier.train(train_data, validation_data, epochs, batch_size)

  return text_classifier


class TextClassifier(classification_model.ClassificationModel):
  """TextClassifier class for inference and exporting to tflite."""

  def __init__(self,
               model_export_format,
               model_spec,
               index_to_label,
               num_classes,
               shuffle=True):
    """Init function for TextClassifier class.

    Args:
      model_export_format: Model export format such as saved_model / tflite.
      model_spec: Specification for the model.
      index_to_label: A list that map from index to label class name.
      num_classes: Number of label classes.
      shuffle: Whether the data should be shuffled.
    """
    super(TextClassifier, self).__init__(
        model_export_format,
        model_spec,
        index_to_label,
        num_classes,
        shuffle,
        train_whole_model=True)

  def preprocess(self, raw_text, label):
    """Preprocess the text."""
    # TODO(yuqili): remove this method once preprocess for image classifier is
    # also moved to DataLoader part.
    return raw_text, label

  def get_dataset_fn(self, input_data, global_batch_size, is_training):
    """Gets a closure to create a dataset."""

    def _dataset_fn(ctx=None):
      """Returns tf.data.Dataset for text classifier retraining."""
      batch_size = ctx.get_per_replica_batch_size(
          global_batch_size) if ctx else global_batch_size
      dataset = self._gen_dataset(
          input_data,
          batch_size,
          is_training=is_training,
          input_pipeline_context=ctx)
      return dataset

    return _dataset_fn

  def train(self, train_data, validation_data=None, epochs=None, batch_size=32):
    """Feeds the training data for training."""

    train_input_fn = self.get_dataset_fn(
        train_data, batch_size, is_training=True)

    validation_steps = 0
    validation_input_fn = None
    if validation_data is not None:
      validation_input_fn = self.get_dataset_fn(
          validation_data, batch_size, is_training=False)
      validation_steps = validation_data.size // batch_size

    steps_per_epoch = train_data.size // batch_size

    self.model = self.model_spec.run_classifier(train_input_fn,
                                                validation_input_fn, epochs,
                                                steps_per_epoch,
                                                validation_steps,
                                                self.num_classes)

    return self.model

  def export(self,
             tflite_filename,
             label_filename,
             vocab_filename,
             quantized=False,
             quantization_steps=None,
             representative_data=None):
    """Converts the retrained model based on `model_export_format`.

    Args:
      tflite_filename: File name to save tflite model.
      label_filename: File name to save labels.
      vocab_filename: File name to save vocabulary.
      quantized: boolean, if True, save quantized model.
      quantization_steps: Number of post-training quantization calibration steps
        to run. Used only if `quantized` is True.
      representative_data: Representative data used for post-training
        quantization. Used only if `quantized` is True.
    """
    if self.model_export_format != mef.ModelExportFormat.TFLITE:
      raise ValueError('Model export format %s is not supported currently.' %
                       self.model_export_format)
    self.model_spec.set_shape(self.model)
    self._export_tflite(tflite_filename, label_filename, quantized,
                        quantization_steps, representative_data)

    self.model_spec.save_vocab(vocab_filename)
