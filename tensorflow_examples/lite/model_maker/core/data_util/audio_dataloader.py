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
"""Audio dataloader."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random

import librosa
import tensorflow as tf
from tensorflow_examples.lite.model_maker.core.data_util import dataloader
from tensorflow_examples.lite.model_maker.core.task.model_spec import audio_spec


class ExamplesHelper(object):
  """Helper class for matching examples and labels."""

  @classmethod
  def from_examples_folder(cls, path, examples_filter_fn):
    """Helper function for loading examples and parsing labels from example path.

    This path contain a number of folders, each named by the category name. Each
    folder contain a number of files. This helper class loads and parse the
    tree structure.

    Example folder:
      /category1
        /file1.wav
        /file2.wav
      /category2
        /file2.wav
        /README

    Usage:
    >>> helper = ExamplesHelper.from_example_folder(path, is_wav)
    >>> # helper.shuffle() if shuffle is needed
    >>> helper.examples_and_labels()
    ('/category1/file1.wav', '/category1/file2.wav', '/category2/file2.wav'),
    ('category1', 'category1', 'category2')
    >>> helper.index_to_label()
    ('category1', 'category2')
    >>> helper.examples_and_label_indices()
    ('/category1/file1.wav', '/category1/file2.wav', '/category2/file2.wav'),
    (0, 0, 1)

    Args:
      path: String, relative path to the data folder. This folder should contain
        a list of sub-folders, named after its categories.
      examples_filter_fn: A lambda function to filter out unrelated files. It
        takes in a full path to the example file and returns a boolean,
        representing if this example file can be preserved.

    Returns:
      An instance of ExamplesHelper.
    """

    def _list_files(path):
      return tf.io.gfile.glob(os.path.join(path, '*', '*'))

    def _get_label(example):
      """Parses the example path and return the label string."""
      return example.rsplit(os.path.sep, 2)[1]

    examples = list(filter(examples_filter_fn, _list_files(path)))
    labels = list(map(_get_label, examples))
    return cls(examples, labels)

  def __init__(self, examples, labels):
    self.index_to_label = sorted(list(set(labels)))  # [label]
    self.label_to_index = {
        label: i for i, label in enumerate(self.index_to_label)
    }
    self._data = sorted(list(zip(examples, labels)))  # [(example, label)]

  def shuffle(self):
    random.shuffle(self._data)

  def examples_and_label_indices(self):
    """Returns a tuple of example and a tuple of their corresponding label idx."""
    examples, labels = self.examples_and_labels()
    label_indicies = tuple(map(self.label_to_index.get, labels))
    return examples, label_indicies

  def examples_and_labels(self):
    """Returns a tuple of example and a tuple of their corresponding labels."""
    if not self._data:
      return (), ()
    examples, labels = zip(*self._data)
    return examples, labels

  def examples_and_label_indices_ds(self):
    examples, labels = self.examples_and_label_indices()
    wav_ds = tf.data.Dataset.from_tensor_slices(list(examples))
    label_ds = tf.data.Dataset.from_tensor_slices(list(labels))
    ds = tf.data.Dataset.zip((wav_ds, label_ds))
    return ds


class DataLoader(dataloader.ClassificationDataLoader):
  """DataLoader for audio tasks."""

  def __init__(self, dataset, size, index_to_label, spec):
    super(DataLoader, self).__init__(dataset, size, index_to_label)
    self._spec = spec

  def split(self, fraction):
    return self._split(fraction, self.index_to_label, self._spec)

  @classmethod
  def from_folder(cls, spec, data_path, is_training=True, shuffle=True):
    """Load audio files from a data_path.

    - The root `data_path` folder contains a number of folders. The name for
    each folder is the name of the audio class.

    - Within each folder, there are a number of .wav files. Each .wav file
    corresponds to an example. Each .wav file is mono (single-channel) and has
    the typical 16 bit pulse-code modulation (PCM) encoding.

    - .wav files will be resampled to spec.target_sample_rate then fed into
    spec.preprocess_ds for split and other operations. Normally long wav files
    will be split into multiple snippets. And wav files shorter than a certain
    threshold will be ignored.

    Args:
      spec: instance of audio_spec.BaseSpec.
      data_path: string, location to the audio files.
      is_training: boolean, if True, apply training augmentation to the dataset.
      shuffle: boolean, if True, random shuffle data.

    Returns:
      AudioDataLoader containing audio spectrogram (or any data type generated
      by spec.preprocess_ds) and labels.
    """
    assert isinstance(spec, audio_spec.BaseSpec)
    root_dir = os.path.abspath(data_path)
    helper = ExamplesHelper.from_examples_folder(root_dir,
                                                 lambda s: s.endswith('.wav'))
    if shuffle:
      helper.shuffle()
    ds = helper.examples_and_label_indices_ds()

    if len(ds) == 0:  # pylint: disable=g-explicit-length-test
      raise ValueError('No audio files found.')

    return DataLoader(ds, len(ds), helper.index_to_label, spec)

  def gen_dataset(self,
                  batch_size=1,
                  is_training=False,
                  shuffle=False,
                  input_pipeline_context=None,
                  preprocess=None,
                  drop_remainder=False):
    """Generate a shared and batched tf.data.Dataset for training/evaluation.

    Args:
      batch_size: A integer, the returned dataset will be batched by this size.
      is_training: A boolean, when True, the returned dataset will be optionally
        shuffled and repeated as an endless dataset.
      shuffle: A boolean, when True, the returned dataset will be shuffled to
        create randomness during model training.
      input_pipeline_context: A InputContext instance, used to shared dataset
        among multiple workers when distribution strategy is used.
      preprocess: A function taking three arguments in order, feature, label and
        boolean is_training.
      drop_remainder: boolean, whether the finaly batch drops remainder.

    Returns:
      A TF dataset ready to be consumed by Keras model.
    """
    # This argument is only used for image dataset for now. Audio preprocessing
    # is defined in the spec.
    del preprocess
    ds = self._dataset
    spec = self._spec
    autotune = tf.data.AUTOTUNE

    options = tf.data.Options()
    options.experimental_deterministic = False
    ds = ds.with_options(options)

    ds = dataloader.shard(ds, input_pipeline_context)

    @tf.function
    def _load_wav(filepath, label):
      file_contents = tf.io.read_file(filepath)
      # shape: (audio_samples, 1), dtype: float32
      wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
      # shape: (audio_samples,)
      wav = tf.squeeze(wav, axis=-1)
      return wav, sample_rate, label

    # This is a eager mode numpy_function. It can be converted to a tf.function
    # using https://www.tensorflow.org/io/api_docs/python/tfio/audio/resample
    def _resample_numpy(waveform, sample_rate, label):
      waveform = librosa.resample(
          waveform, orig_sr=sample_rate, target_sr=spec.target_sample_rate)
      return waveform, label

    @tf.function
    def _resample(waveform, sample_rate, label):
      # Short circuit resampling if possible.
      if sample_rate == spec.target_sample_rate:
        return [waveform, label]
      return tf.numpy_function(
          _resample_numpy,
          inp=(waveform, sample_rate, label),
          Tout=[tf.float32, tf.int32])

    @tf.function
    def _elements_finite(preprocess_data, unused_label):
      # Make sure that the data sent to the model does not contain nan or inf
      # values. This should be the last filter applied to the dataset.
      # Arguably we could possibly apply this filter to all tasks.
      return tf.size(preprocess_data) > 0 and tf.math.reduce_all(
          tf.math.is_finite(preprocess_data))

    ds = ds.map(_load_wav, num_parallel_calls=autotune)
    ds = ds.map(_resample, num_parallel_calls=autotune)
    ds = spec.preprocess_ds(ds, is_training=is_training)
    ds = ds.filter(_elements_finite)

    if is_training:
      if shuffle:
        # Shuffle size should be bigger than the batch_size. Otherwise it's only
        # shuffling within the batch, which equals to not having shuffle.
        buffer_size = 3 * batch_size
        # But since we are doing shuffle before repeat, it doesn't make sense to
        # shuffle more than total available entries.
        # TODO(wangtz): Do we want to do shuffle before / after repeat?
        # Shuffle after repeat will give a more randomized dataset and mix the
        # epoch boundary: https://www.tensorflow.org/guide/data
        ds = ds.shuffle(buffer_size=min(self._size, buffer_size))

    ds = ds.batch(batch_size, drop_remainder=drop_remainder)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    # TODO(b/171449557): Consider converting ds to distributed ds here.
    return ds

  @classmethod
  def from_esc50(cls, spec, data_path):
    """Load ESC50 style audio samples.

    ESC50 file structure is expalined in https://github.com/karolpiczak/ESC-50
    Audio files should be put in ${data_path}/audio
    Metadata file should be put in ${data_path}/meta/esc50.csv

    Note that only YAMNet model is supported.

    Args:
      spec: An instance of audio_spec.YAMNet
      data_path: string, location to the audio files.

    Returns:
      An instance of AudioDataLoader containing audio samples and labels.
    """
    # TODO(b/178083096): remove this restriction.
    assert isinstance(spec, audio_spec.YAMNetSpec)
    return NotImplemented
