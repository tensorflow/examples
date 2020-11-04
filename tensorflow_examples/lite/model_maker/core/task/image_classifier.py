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
"""ImageClassier class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile

import tensorflow.compat.v2 as tf

from tensorflow_examples.lite.model_maker.core.task import metadata_writer_for_image_classifier as metadata_writer

from tensorflow_examples.lite.model_maker.core import compat
from tensorflow_examples.lite.model_maker.core.task import classification_model
from tensorflow_examples.lite.model_maker.core.task import hub_loader
from tensorflow_examples.lite.model_maker.core.task import image_preprocessing
from tensorflow_examples.lite.model_maker.core.task import model_spec as ms
from tensorflow_examples.lite.model_maker.core.task import model_util
from tensorflow_examples.lite.model_maker.core.task import train_image_classifier_lib

from tensorflow_hub.tools.make_image_classifier import make_image_classifier_lib as hub_lib
from tflite_support import metadata as _metadata  # pylint: disable=g-direct-tensorflow-import


def get_hub_lib_hparams(**kwargs):
  """Gets the hyperparameters for the tensorflow hub's library."""
  hparams = hub_lib.get_default_hparams()
  return train_image_classifier_lib.add_params(hparams, **kwargs)


def create(train_data,
           model_spec='efficientnet_lite0',
           validation_data=None,
           batch_size=None,
           epochs=None,
           train_whole_model=None,
           dropout_rate=None,
           learning_rate=None,
           momentum=None,
           shuffle=False,
           use_augmentation=False,
           use_hub_library=True,
           warmup_steps=None,
           model_dir=None,
           do_train=True):
  """Loads data and retrains the model based on data for image classification.

  Args:
    train_data: Training data.
    model_spec: Specification for the model.
    validation_data: Validation data. If None, skips validation process.
    batch_size: Number of samples per training step. If `use_hub_library` is
      False, it represents the base learning rate when train batch size is 256
      and it's linear to the batch size.
    epochs: Number of epochs for training.
    train_whole_model: If true, the Hub module is trained together with the
      classification layer on top. Otherwise, only train the top classification
      layer.
    dropout_rate: The rate for dropout.
    learning_rate: Base learning rate when train batch size is 256. Linear to
      the batch size.
    momentum: a Python float forwarded to the optimizer. Only used when
      `use_hub_library` is True.
    shuffle: Whether the data should be shuffled.
    use_augmentation: Use data augmentation for preprocessing.
    use_hub_library: Use `make_image_classifier_lib` from tensorflow hub to
      retrain the model.
    warmup_steps: Number of warmup steps for warmup schedule on learning rate.
      If None, the default warmup_steps is used which is the total training
      steps in two epochs. Only used when `use_hub_library` is False.
    model_dir: The location of the model checkpoint files. Only used when
      `use_hub_library` is False.
    do_train: Whether to run training.

  Returns:
    An instance of ImageClassifier class.
  """
  model_spec = ms.get(model_spec)
  if compat.get_tf_behavior() not in model_spec.compat_tf_versions:
    raise ValueError('Incompatible versions. Expect {}, but got {}.'.format(
        model_spec.compat_tf_versions, compat.get_tf_behavior()))

  if use_hub_library:
    hparams = get_hub_lib_hparams(
        batch_size=batch_size,
        train_epochs=epochs,
        do_fine_tuning=train_whole_model,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate,
        momentum=momentum)
  else:
    hparams = train_image_classifier_lib.HParams.get_hparams(
        batch_size=batch_size,
        train_epochs=epochs,
        do_fine_tuning=train_whole_model,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        model_dir=model_dir)

  image_classifier = ImageClassifier(
      model_spec,
      train_data.index_to_label,
      train_data.num_classes,
      shuffle=shuffle,
      hparams=hparams,
      use_augmentation=use_augmentation)

  if do_train:
    tf.compat.v1.logging.info('Retraining the models...')
    image_classifier.train(train_data, validation_data)
  else:
    # Used in evaluation.
    image_classifier.create_model(with_loss_and_metrics=True)

  return image_classifier


def _get_model_info(model_spec,
                    num_classes,
                    quantization_config=None,
                    version='v1'):
  """Gets the specific info for the image model."""

  if not isinstance(model_spec, ms.ImageModelSpec):
    raise ValueError('Currently only support models for image classification.')

  image_min = 0
  image_max = 1

  name = model_spec.name
  if quantization_config:
    name += '_quantized'
    if quantization_config.inference_input_type == tf.uint8:
      image_min = 0
      image_max = 255
    elif quantization_config.inference_input_type == tf.int8:
      image_min = -128
      image_max = 127

  return metadata_writer.ModelSpecificInfo(
      model_spec.name,
      version,
      image_width=model_spec.input_image_shape[1],
      image_height=model_spec.input_image_shape[0],
      mean=model_spec.mean_rgb,
      std=model_spec.stddev_rgb,
      image_min=image_min,
      image_max=image_max,
      num_classes=num_classes)


class ImageClassifier(classification_model.ClassificationModel):
  """ImageClassifier class for inference and exporting to tflite."""

  def __init__(self,
               model_spec,
               index_to_label,
               num_classes,
               shuffle=True,
               hparams=hub_lib.get_default_hparams(),
               use_augmentation=False):
    """Init function for ImageClassifier class.

    Args:
      model_spec: Specification for the model.
      index_to_label: A list that map from index to label class name.
      num_classes: Number of label classes.
      shuffle: Whether the data should be shuffled.
      hparams: A namedtuple of hyperparameters. This function expects
        .dropout_rate: The fraction of the input units to drop, used in dropout
          layer.
        .do_fine_tuning: If true, the Hub module is trained together with the
          classification layer on top.
      use_augmentation: Use data augmentation for preprocessing.
    """
    super(ImageClassifier,
          self).__init__(model_spec, index_to_label, num_classes, shuffle,
                         hparams.do_fine_tuning)
    self._hparams = hparams
    self._preprocessor = image_preprocessing.Preprocessor(
        self.model_spec.input_image_shape,
        num_classes,
        self.model_spec.mean_rgb,
        self.model_spec.stddev_rgb,
        use_augmentation=use_augmentation)
    self.history = None  # Training history that returns from `keras_model.fit`.

  def preprocess(self, sample, label, is_training=False):
    return self._preprocessor(sample, label, is_training)

  def _get_tflite_input_tensors(self, input_tensors):
    """Gets the input tensors for the TFLite model."""
    return input_tensors

  def create_model(self, hparams=None, with_loss_and_metrics=False):
    """Creates the classifier model for retraining."""
    hparams = self._get_hparams_or_default(hparams)

    module_layer = hub_loader.HubKerasLayerV1V2(
        self.model_spec.uri, trainable=hparams.do_fine_tuning)
    self.model = hub_lib.build_model(module_layer, hparams,
                                     self.model_spec.input_image_shape,
                                     self.num_classes)
    if with_loss_and_metrics:
      # Adds loss and metrics in the keras model.
      self.model.compile(
          loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
          metrics=['accuracy'])

  def train(self, train_data, validation_data=None, hparams=None):
    """Feeds the training data for training.

    Args:
      train_data: Training data.
      validation_data: Validation data. If None, skips validation process.
      hparams: An instance of hub_lib.HParams or
        train_image_classifier_lib.HParams. Anamedtuple of hyperparameters.

    Returns:
      The tf.keras.callbacks.History object returned by tf.keras.Model.fit*().
    """
    self.create_model()
    hparams = self._get_hparams_or_default(hparams)

    if train_data.size < hparams.batch_size:
      raise ValueError('The size of the train_data (%d) couldn\'t be smaller '
                       'than batch_size (%d). To solve this problem, set '
                       'the batch_size smaller or increase the size of the '
                       'train_data.' % (train_data.size, hparams.batch_size))

    # TODO(b/171449557): Consider refactoring this code
    def train_preprocesor(image, label):
      return self._preprocessor(image, label, is_training=True)

    train_ds = train_data.gen_dataset(
        hparams.batch_size,
        is_training=True,
        shuffle=self.shuffle,
        preprocess=train_preprocesor)
    train_data_and_size = (train_ds, train_data.size)

    validation_ds = None
    validation_size = 0
    if validation_data is not None:
      validation_ds = validation_data.gen_dataset(
          hparams.batch_size, is_training=False, preprocess=self.preprocess)
      validation_size = validation_data.size
    validation_data_and_size = (validation_ds, validation_size)

    # Trains the models.
    lib = hub_lib
    if isinstance(hparams, train_image_classifier_lib.HParams):
      lib = train_image_classifier_lib
    self.history = lib.train_model(self.model, hparams, train_data_and_size,
                                   validation_data_and_size)
    return self.history

  def export(self,
             export_dir,
             tflite_filename='model.tflite',
             label_filename='labels.txt',
             saved_model_filename='saved_model',
             export_format=None,
             **kwargs):
    """Converts the retrained model based on `model_export_format`.

    Args:
      export_dir: The directory to save exported files.
      tflite_filename: File name to save tflite model. The full export path is
        {export_dir}/{tflite_filename}.
      label_filename: File name to save labels. The full export path is
        {export_dir}/{label_filename}.
      saved_model_filename: Path to SavedModel or H5 file to save the model. The
        full export path is
        {export_dir}/{saved_model_filename}/{saved_model.pb|assets|variables}.
      export_format: List of export format that could be saved_model, tflite,
        label.
      **kwargs: Other parameters like `quantized` for TFLITE model.
    """
    super(ImageClassifier, self).export(
        export_dir,
        tflite_filename=tflite_filename,
        label_filename=label_filename,
        saved_model_filename=saved_model_filename,
        export_format=export_format,
        **kwargs)

  def _export_tflite(self,
                     tflite_filepath,
                     quantization_config=None,
                     with_metadata=True,
                     export_metadata_json_file=False):
    """Converts the retrained model to tflite format and saves it.

    Args:
      tflite_filepath: File path to save tflite model.
      quantization_config: Configuration for post-training quantization.
      with_metadata: Whether the output tflite model contains metadata.
      export_metadata_json_file: Whether to export metadata in json file. If
        True, export the metadata in the same directory as tflite model.Used
        only if `with_metadata` is True.
    """
    model_util.export_tflite(
        self.model,
        tflite_filepath,
        quantization_config,
        preprocess=self.preprocess)
    if with_metadata:
      with tempfile.TemporaryDirectory() as temp_dir:
        tf.compat.v1.logging.info(
            'Label file is inside the TFLite model with metadata.')
        label_filepath = os.path.join(temp_dir, 'labels.txt')
        self._export_labels(label_filepath)

        model_info = _get_model_info(
            self.model_spec,
            self.num_classes,
            quantization_config=quantization_config)
        # Generate the metadata objects and put them in the model file
        populator = metadata_writer.MetadataPopulatorForImageClassifier(
            tflite_filepath, model_info, label_filepath)
        populator.populate()

      # Validate the output model file by reading the metadata and produce
      # a json file with the metadata under the export path
      if export_metadata_json_file:
        displayer = _metadata.MetadataDisplayer.with_model_file(tflite_filepath)
        export_json_file = os.path.splitext(tflite_filepath)[0] + '.json'

        content = displayer.get_metadata_json()
        with open(export_json_file, 'w') as f:
          f.write(content)

  def _get_hparams_or_default(self, hparams):
    """Returns hparams if not none, otherwise uses default one."""
    return hparams if hparams else self._hparams
