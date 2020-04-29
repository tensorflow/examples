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

import tensorflow.compat.v2 as tf

from tensorflow_examples.lite.model_maker.core import compat
from tensorflow_examples.lite.model_maker.core.export_format import ExportFormat
from tensorflow_examples.lite.model_maker.core.task import classification_model
from tensorflow_examples.lite.model_maker.core.task import hub_loader
from tensorflow_examples.lite.model_maker.core.task import image_preprocessing
from tensorflow_examples.lite.model_maker.core.task import metadata
from tensorflow_examples.lite.model_maker.core.task import model_spec as ms
from tensorflow_examples.lite.model_maker.core.task import train_image_classifier_lib

from tensorflow_hub.tools.make_image_classifier import make_image_classifier_lib as hub_lib


def get_hub_lib_hparams(**kwargs):
  """Gets the hyperparameters for the tensorflow hub's library."""
  hparams = hub_lib.get_default_hparams()
  return train_image_classifier_lib.add_params(hparams, **kwargs)


def create(train_data,
           model_spec=ms.efficientnet_lite0_spec,
           shuffle=False,
           validation_data=None,
           batch_size=None,
           epochs=None,
           train_whole_model=None,
           dropout_rate=None,
           learning_rate=None,
           momentum=None,
           use_augmentation=False,
           use_hub_library=True,
           warmup_steps=None,
           model_dir=None):
  """Loads data and retrains the model based on data for image classification.

  Args:
    train_data: Training data.
    model_spec: Specification for the model.
    shuffle: Whether the data should be shuffled.
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
    use_augmentation: Use data augmentation for preprocessing.
    use_hub_library: Use `make_image_classifier_lib` from tensorflow hub to
      retrain the model.
    warmup_steps: Number of warmup steps for warmup schedule on learning rate.
      If None, the default warmup_steps is used which is the total training
      steps in two epochs. Only used when `use_hub_library` is False.
    model_dir: The location of the model checkpoint files. Only used when
      `use_hub_library` is False.

  Returns:
    An instance of ImageClassifier class.
  """
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

  tf.compat.v1.logging.info('Retraining the models...')
  image_classifier.train(train_data, validation_data)

  return image_classifier


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
    self.hparams = hparams
    self.model = self._create_model()
    self.preprocessor = image_preprocessing.Preprocessor(
        self.model_spec.input_image_shape,
        num_classes,
        self.model_spec.mean_rgb,
        self.model_spec.stddev_rgb,
        use_augmentation=use_augmentation)
    self.history = None  # Training history that returns from `keras_model.fit`.

  def _create_model(self, hparams=None):
    """Creates the classifier model for retraining."""
    hparams = self._get_hparams_or_default(hparams)

    module_layer = hub_loader.HubKerasLayerV1V2(
        self.model_spec.uri, trainable=hparams.do_fine_tuning)
    return hub_lib.build_model(module_layer, hparams,
                               self.model_spec.input_image_shape,
                               self.num_classes)

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
    hparams = self._get_hparams_or_default(hparams)

    train_ds = self._gen_dataset(
        train_data, hparams.batch_size, is_training=True)
    train_data_and_size = (train_ds, train_data.size)

    validation_ds = None
    validation_size = 0
    if validation_data is not None:
      validation_ds = self._gen_dataset(
          validation_data, hparams.batch_size, is_training=False)
      validation_size = validation_data.size
    validation_data_and_size = (validation_ds, validation_size)

    # Trains the models.
    lib = hub_lib
    if isinstance(hparams, train_image_classifier_lib.HParams):
      lib = train_image_classifier_lib
    self.history = lib.train_model(self.model, hparams, train_data_and_size,
                                   validation_data_and_size)
    return self.history

  def preprocess(self, image, label, is_training=False):
    return self.preprocessor(image, label, is_training)

  def _gen_dataset(self, data, batch_size=32, is_training=True):
    """Generates training / validation dataset."""
    ds = data.dataset
    ds = ds.map(lambda image, label: self.preprocess(image, label, is_training))

    if is_training:
      if self.shuffle:
        ds = ds.shuffle(buffer_size=min(data.size, 100))
      ds = ds.repeat()

    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds

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
    # Default export formats are TFLite models and labels.
    if export_format is None:
      export_format = [ExportFormat.TFLITE, ExportFormat.LABEL]
    if not isinstance(export_format, list):
      export_format = [export_format]

    if not tf.io.gfile.exists(export_dir):
      tf.io.gfile.makedirs(export_dir)

    label_filepath = None
    if ExportFormat.LABEL in export_format:
      label_filepath = os.path.join(export_dir, label_filename)
      self._export_labels(label_filepath)

    if ExportFormat.TFLITE in export_format:
      tflite_filepath = os.path.join(export_dir, tflite_filename)
      self._export_tflite(tflite_filepath, label_filepath, **kwargs)

    if ExportFormat.SAVED_MODEL in export_format:
      saved_model_filepath = os.path.join(export_dir, saved_model_filename)
      self._export_saved_model(saved_model_filepath, **kwargs)

  def _export_tflite(self,
                     tflite_filepath,
                     label_filepath,
                     quantized=False,
                     quantization_steps=None,
                     representative_data=None,
                     inference_input_type=tf.float32,
                     inference_output_type=tf.float32,
                     with_metadata=False,
                     export_metadata_json_file=False):
    """Converts the retrained model to tflite format and saves it.


    Args:
      tflite_filepath: File path to save tflite model.
      label_filepath: File path to save labels.
      quantized: boolean, if True, save quantized model.
      quantization_steps: Number of post-training quantization calibration steps
        to run. Used only if `quantized` is True.
      representative_data: Representative data used for post-training
        quantization. Used only if `quantized` is True.
      inference_input_type: Target data type of real-number input arrays. Allows
        for a different type for input arrays. Defaults to tf.float32. Must be
        be `{tf.float32, tf.uint8, tf.int8}`
      inference_output_type: Target data type of real-number output arrays.
        Allows for a different type for output arrays. Defaults to tf.float32.
         Must be `{tf.float32, tf.uint8, tf.int8}`
      with_metadata: Whether the output tflite model contains metadata.
      export_metadata_json_file: Whether to export metadata in json file. If
        True, export the metadata in the same directory as tflite model.Used
        only if `with_metadata` is True.
    """
    super(ImageClassifier,
          self)._export_tflite(tflite_filepath, quantized, quantization_steps,
                               representative_data, inference_input_type,
                               inference_output_type)
    if with_metadata:
      if not metadata.TFLITE_SUPPORT_TOOLS_INSTALLED:
        tf.compat.v1.logging.warning('Needs to install tflite-support package.')
        return

      if label_filepath is None:
        tf.compat.v1.logging.warning(
            'Label filepath is needed when exporting TFLite with metadata.')
        return

      model_info = metadata.get_model_info(self.model_spec, quantized=quantized)
      populator = metadata.MetadataPopulatorForImageClassifier(
          tflite_filepath, model_info, label_filepath)
      populator.populate()

      if export_metadata_json_file:
        metadata.export_metadata_json_file(tflite_filepath)

  def _get_hparams_or_default(self, hparams):
    """Returns hparams if not none, otherwise uses default one."""
    return hparams if hparams else self.hparams
