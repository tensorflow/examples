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

import bisect
import os
import random

import librosa
import numpy as np
from scipy.io import wavfile
import tensorflow as tf
from tensorflow_examples.lite.model_maker.core.data_util import dataloader
from tensorflow_examples.lite.model_maker.core.task.model_spec import audio_spec


def _list_files(path):
  return tf.io.gfile.glob(path + r'/*/*')


class ExamplesHelper(object):
  """Helper class for loading examples and parsing labels from example path.

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
  >>> helper = ExamplesHelper(path, is_wav)
  >>> # helper.shuffle() if shuffle is needed
  >>> helper.examples_and_labels()
  ('/category1/file1.wav', '/category1/file2.wav', '/category2/file2.wav'),
  ('category1', 'category1', 'category2')
  >>> helper.sorted_cateogries()
  ('category1', 'category2')
  >>> helper.examples_and_label_indices()
  ('/category1/file1.wav', '/category1/file2.wav', '/category2/file2.wav'), (0,
  0, 1)
  """

  def __init__(self, path, filter_fn=None):

    def _is_dir(folder):
      return tf.io.gfile.isdir(os.path.join(path, folder))

    if not filter_fn:
      filter_fn = lambda x: True
    # Immutable after `__init__`
    self._path = path
    self._sorted_examples = sorted(filter(filter_fn, _list_files(path)))
    self._sorted_categories = sorted(filter(_is_dir, tf.io.gfile.listdir(path)))
    # Mutable only by `shuffle` method
    self._examples = self._sorted_examples

  @property
  def sorted_cateogries(self):
    return self._sorted_categories

  def shuffle(self):
    return random.shuffle(self._examples)

  def _get_label(self, example):
    """Parses the example path and return the label string."""
    return example.rsplit(os.path.sep, 2)[1]

  def _category_idx(self, label):
    """Converts label string to index."""
    idx = bisect.bisect_left(self._sorted_categories, label)
    if idx != len(
        self._sorted_categories) and self._sorted_categories[idx] == label:
      return idx
    raise ValueError('Unknown label: ', label)

  def examples_and_label_indices(self):
    """Returns a list of example and a list of their corresponding label idx."""
    labels = (self._get_label(example) for example in self._examples)
    return self._examples, [self._category_idx(label) for label in labels]

  def examples_and_labels(self):
    """Returns a list of example and a list of their corresponding labels."""
    labels = [self._get_label(example) for example in self._examples]
    return self._examples, labels


def _resample_and_cut(cache_path, spec, example, label):
  """Resample and cut the audio files into snippets."""

  def _new_path(i):
    # Note that `splitext` handles hidden files differently and here we just
    # assume that audio files are not hidden files.
    # os.path.splitext('/root/path/.wav') => ('/root/path/.wav', '')
    # instead of ('/root/path/', '.wav') as this code expects.
    filename_without_extension = os.path.splitext(os.path.basename(example))[0]
    new_filename = filename_without_extension + '_%d.wav' % i
    return os.path.join(cache_path, label, new_filename)

  sampling_rate, xs = wavfile.read(example)
  if xs.dtype != np.int16:
    raise ValueError(
        'DataLoader expects 16 bit PCM encoded WAV files, but {} has type {}'
        .format(example, xs.dtype))

  # Resample.
  if spec.target_sample_rate != sampling_rate:
    # Resample, librosa.resample only works with float32.
    # Ref: https://github.com/bmcfee/resampy/issues/44
    xs = xs.astype(np.float32)
    xs = librosa.resample(
        xs, orig_sr=sampling_rate,
        target_sr=spec.target_sample_rate).astype(np.int16)

  # Extract snippets.
  n_samples_per_snippet = int(spec.snippet_duration_sec *
                              spec.target_sample_rate)
  begin_index = 0
  count = 0
  while begin_index + n_samples_per_snippet <= len(xs):
    snippet = xs[begin_index:begin_index + n_samples_per_snippet]
    wavfile.write(_new_path(count), spec.target_sample_rate, snippet)
    begin_index += n_samples_per_snippet
    count += 1

  return count


class DataLoader(dataloader.ClassificationDataLoader):
  """DataLoader for audio tasks."""

  def __init__(self, dataset, size, index_to_label, spec):
    super(DataLoader, self).__init__(dataset, size, index_to_label)
    self._spec = spec

  def split(self, fraction):
    return self._split(fraction, self.index_to_label, self._spec)

  @classmethod
  def _has_cache(cls, cache_path):
    if not tf.io.gfile.exists(cache_path):
      return False
    cache_examples = _list_files(cache_path)
    return len(cache_examples) > 1

  @classmethod
  def _create_cache(cls, spec, src_path, cache_path):
    """Resample and extract audio snippets in wav format under `cache_path`."""
    os.makedirs(cache_path, exist_ok=True)

    # List all .wav files.
    helper = ExamplesHelper(src_path, lambda s: s.endswith('.wav'))
    examples, labels = helper.examples_and_labels()
    total_samples = 0

    print('Processing audio files:')
    bar = tf.keras.utils.Progbar(len(labels), unit_name='file')

    for example, label in zip(examples, labels):
      bar.add(1)
      os.makedirs(os.path.join(cache_path, label), exist_ok=True)
      total_samples += _resample_and_cut(cache_path, spec, example, label)

    return total_samples

  @classmethod
  def _from_cache(cls, spec, cache_path, is_training, shuffle):
    helper = ExamplesHelper(cache_path)
    if shuffle:
      helper.shuffle()
    examples, labels = helper.examples_and_label_indices()
    index_to_labels = helper.sorted_cateogries

    path_ds = tf.data.Dataset.from_tensor_slices(examples)
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    ds = tf.data.Dataset.zip((path_ds, label_ds))

    tf.compat.v1.logging.info('Loaded %d audio files.', len(ds))
    return DataLoader(ds, len(ds), index_to_labels, spec)

  @classmethod
  def from_folder(cls, spec, data_path, is_training=True, shuffle=True):
    """Load audio files from a data_path.

    - The root `data_path` folder contains a number of folders. The name for
    each folder is the name of the audio class.

    - Within each folder, there are a number of .wav files. Each .wav file
    corresponds to an example. Each .wav file is mono (single-channel) and has
    the typical 16 bit pulse-code modulation (PCM) encoding.

    - .wav files are expected to be spec.snippet_duration_sec long. DataLoader
    ignores files with shorter duration and extracts snippets from .wav files
    with longer duration.

    Args:
      spec: instance of audio_spec.BaseSpec.
      data_path: string, location to the audio files.
      is_training: boolean, if True, apply training augmentation to the dataset.
      shuffle: boolean, if True, random shuffle data.

    Returns:
      AudioDataLoader containing audio spectrogram (or any data type returned by
      spec.preprocess) and labels.
    """
    assert isinstance(spec, audio_spec.BaseSpec)
    root_dir = os.path.abspath(data_path)
    cache_dir = os.path.join(root_dir, 'cache')

    if not cls._has_cache(cache_dir):
      cnt = cls._create_cache(spec, data_path, cache_dir)
      if cnt == 0:
        raise ValueError('No audio files found.')
      print('Cached {} audio samples.'.format(cnt))
    return cls._from_cache(spec, cache_dir, is_training, shuffle)

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
    _ = preprocess
    ds = self._dataset
    spec = self._spec
    autotune = tf.data.AUTOTUNE

    ds = dataloader.shard(ds, input_pipeline_context)

    @tf.function
    def _load_wav(filepath, label):
      file_contents = tf.io.read_file(filepath)
      # shape: (target_sample_rate, 1)
      wav, _ = tf.audio.decode_wav(file_contents, desired_channels=1)
      # shape: (target_sample_rate,)
      wav = tf.squeeze(wav, axis=-1)
      # shape: (1, target_sample_rate)
      wav = tf.expand_dims(wav, 0)
      return wav, label

    @tf.function
    def _crop(waveform, label):
      # shape: (1, expected_waveform_len)
      cropped = tf.slice(
          waveform, begin=[0, 0], size=[1, spec.expected_waveform_len])
      return cropped, label

    @tf.function
    def _elements_finite(preprocess_data, unused_label):
      # Make sure that the data sent to the model does not contain nan or inf
      # values. This should be the last filter applied to the dataset.
      # Arguably we could possibly apply this filter to all tasks.
      return tf.math.reduce_all(tf.math.is_finite(preprocess_data))

    ds = ds.map(_load_wav, num_parallel_calls=autotune)
    ds = ds.map(_crop, num_parallel_calls=autotune)
    ds = ds.map(spec.preprocess, num_parallel_calls=autotune)
    if is_training:
      ds = ds.map(spec.data_augmentation, num_parallel_calls=autotune)
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
