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
"""Audio model specification."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import os
import tempfile

import tensorflow as tf
from tensorflow_examples.lite.model_maker.core.task import model_util


class BaseSpec(abc.ABC):
  """Base model spec for audio classification."""

  compat_tf_versions = (2,)

  def __init__(self, model_dir=None, strategy=None):
    self.model_dir = model_dir
    if not model_dir:
      self.model_dir = tempfile.mkdtemp()
    tf.compat.v1.logging.info('Checkpoints are stored in %s', self.model_dir)
    self.strategy = strategy or tf.distribute.get_strategy()

  @abc.abstractproperty
  def target_sample_rate(self):
    pass

  @abc.abstractmethod
  def create_model(self, num_classes):
    pass

  @abc.abstractmethod
  def run_classifier(self, model, train_input_fn, validation_input_fn, epochs,
                     steps_per_epoch, validation_steps):
    pass

  # Default dummy augmentation that will be applied to train samples.
  @tf.function
  def data_augmentation(self, x, label):
    return x, label

  # Default dummy preprocessing that will be applied to all data samples.
  @tf.function
  def preprocess(self, x, label):
    return x, label


def _remove_suffix_if_possible(text, suffix):
  return text.rsplit(suffix, 1)[0]


TFJS_MODEL_ROOT = 'https://storage.googleapis.com/tfjs-models/tfjs'


def _load_browser_fft_preprocess_model():
  """Load a model replicating WebAudio's AnalyzerNode.getFloatFrequencyData."""
  model_name = 'sc_preproc_model'
  file_extension = '.tar.gz'
  filename = model_name + file_extension
  # Load the preprocessing model, which transforms audio waveform into
  # spectrograms (2D image-like representation of sound).
  # This model replicates WebAudio's AnalyzerNode.getFloatFrequencyData
  # (https://developer.mozilla.org/en-US/docs/Web/API/AnalyserNode/getFloatFrequencyData).
  # It performs short-time Fourier transform (STFT) using a length-2048 Blackman
  # window. It opeartes on mono audio at the 44100-Hz sample rate.
  filepath = tf.keras.utils.get_file(
      filename,
      f'{TFJS_MODEL_ROOT}/speech-commands/conversion/{filename}',
      cache_subdir='model_maker',
      extract=True)
  model_path = _remove_suffix_if_possible(filepath, file_extension)
  return tf.keras.models.load_model(model_path)


def _load_tfjs_speech_command_model():
  """Download TFJS speech command model for fine-tune."""
  origin_root = f'{TFJS_MODEL_ROOT}/speech-commands/v0.3/browser_fft/18w'
  files_to_download = [
      'metadata.json', 'model.json', 'group1-shard1of2', 'group1-shard2of2'
  ]
  for filename in files_to_download:
    filepath = tf.keras.utils.get_file(
        filename,
        f'{origin_root}/{filename}',
        cache_subdir='model_maker/tfjs-sc-model')
  model_path = os.path.join(os.path.dirname(filepath), 'model.json')
  return model_util.load_tfjs_keras_model(model_path)


class BrowserFFTSpec(BaseSpec):
  """Audio classification model spec using Browswer FFT as preprocessing."""

  def __init__(self, model_dir=None, strategy=None):
    super(BrowserFFTSpec, self).__init__(model_dir, strategy)
    self._preprocess_model = _load_browser_fft_preprocess_model()
    self._tfjs_sc_model = _load_tfjs_speech_command_model()

    self.expected_waveform_len = self._preprocess_model.input_shape[-1]
    self.snippet_duration_sec = 1.

  @property
  def target_sample_rate(self):
    return 44100

  @tf.function
  def preprocess(self, x, label):
    # x has shape (1, expected_waveform_len)
    y = self._preprocess_model(x)
    # y has shape (1, embedding_len)
    y = tf.squeeze(y, axis=0)
    # y has shape (embedding_len,)
    return y, label

  def create_model(self, num_classes):
    if num_classes <= 1:
      raise ValueError(
          'AudioClassifier expects `num_classes` to be greater than 1')
    model = tf.keras.Sequential()
    for layer in self._tfjs_sc_model.layers[:-1]:
      model.add(layer)
    model.add(tf.keras.layers.Dense(units=num_classes, activation='softmax'))
    # Freeze all but the last layer of the model. The last layer will be
    # fine-tuned during transfer learning.
    for layer in model.layers[:-1]:
      layer.trainable = False
    return model

  def run_classifier(self, model, epochs, train_ds, validation_ds, **kwargs):
    model.compile(
        optimizer='sgd',
        loss='sparse_categorical_crossentropy',
        metrics=['acc'])

    hist = model.fit(
        train_ds, validation_data=validation_ds, epochs=epochs, **kwargs)
    return hist

  def export_tflite(self, model, tflite_filepath, quantization_config=None):
    """Converts the retrained model to tflite format and saves it.

    This method overrides the default `CustomModel._export_tflite` method, and
    include the pre-processing in the exported TFLite library since support
    library can't handle audio tasks yet.

    Args:
      model: An instance of the keras classification model to be exported.
      tflite_filepath: File path to save tflite model.
      quantization_config: Configuration for post-training quantization.
    """
    combined = tf.keras.Sequential()
    combined.add(self._preprocess_model)
    combined.add(model)
    # Build the model.
    combined.build([None, self.expected_waveform_len])

    # Sets batch size from None to 1 when converting to tflite.
    model_util.set_batch_size(model, batch_size=1)

    model_util.export_tflite(
        combined,
        tflite_filepath,
        quantization_config,
        supported_ops=(tf.lite.OpsSet.TFLITE_BUILTINS,
                       tf.lite.OpsSet.SELECT_TF_OPS))

    # Sets batch size back to None to support retraining later.
    model_util.set_batch_size(model, batch_size=None)


# TODO(b/178083096): Refactor this class and BaseSpec to have a common interface
class YAMNetSpec:
  pass
