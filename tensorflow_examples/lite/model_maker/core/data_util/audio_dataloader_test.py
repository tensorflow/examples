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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
from scipy.io import wavfile

import tensorflow.compat.v2 as tf
from tensorflow_examples.lite.model_maker.core.data_util import audio_dataloader
from tensorflow_examples.lite.model_maker.core.task.model_spec import audio_spec


def write_file(root, filepath):
  full_path = os.path.join(root, filepath)
  os.makedirs(os.path.dirname(full_path), exist_ok=True)
  with open(full_path, 'w') as f:
    f.write('<content>')


def write_sample(root,
                 category,
                 file_name,
                 sample_rate,
                 duration_sec,
                 value,
                 dtype=np.int16):
  os.makedirs(os.path.join(root, category), exist_ok=True)
  xs = value * np.ones(shape=(int(sample_rate * duration_sec),), dtype=dtype)
  wavfile.write(os.path.join(root, category, file_name), sample_rate, xs)


class MockSpec(audio_spec.BaseSpec):

  def __init__(self, *args, **kwargs):
    super(MockSpec, self).__init__(*args, **kwargs)
    self.expected_waveform_len = 44100

  def create_model(self):
    return None

  def run_classifier(self, *args, **kwargs):
    return None

  @property
  def target_sample_rate(self):
    return 44100

  def preprocess_ds(self, ds, is_training=False):
    _ = is_training

    @tf.function
    def _ensure_length(wav, unused_label):
      return len(wav) >= self.expected_waveform_len

    @tf.function
    def _split(wav, label):
      # wav shape: (audio_samples, )
      chunks = tf.math.floordiv(len(wav), self.expected_waveform_len)
      unused = tf.math.floormod(len(wav), self.expected_waveform_len)
      # Drop unused data
      wav = wav[:len(wav) - unused]
      # Split the audio sample into multiple chunks
      wav = tf.reshape(wav, (chunks, 1, self.expected_waveform_len))

      return wav, tf.repeat(tf.expand_dims(label, 0), len(wav))

    autotune = tf.data.AUTOTUNE
    ds = ds.filter(_ensure_length)
    ds = ds.map(_split, num_parallel_calls=autotune).unbatch()
    return ds


class Base(tf.test.TestCase):

  def _get_folder_path(self, sub_folder_name):
    folder_path = os.path.join(self.get_temp_dir(), sub_folder_name)
    if os.path.exists(folder_path):
      return
    tf.compat.v1.logging.info('Test path: %s', folder_path)
    os.mkdir(folder_path)
    return folder_path


class LoadFromESC50Test(Base):

  def test_spec(self):
    folder_path = self._get_folder_path('test_examples_helper')

    spec = audio_spec.YAMNetSpec()
    audio_dataloader.DataLoader.from_esc50(spec, folder_path)

    spec = audio_spec.BrowserFFTSpec()
    with self.assertRaises(AssertionError):
      audio_dataloader.DataLoader.from_esc50(spec, folder_path)


class LoadFromFolderTest(Base):

  def test_spec(self):
    folder_path = self._get_folder_path('test_examples_helper')
    write_sample(folder_path, 'unknown', '2s.wav', 44100, 2, value=1)

    spec = audio_spec.YAMNetSpec()
    with self.assertRaises(AssertionError):
      audio_dataloader.DataLoader.from_folder(spec, folder_path)

    spec = audio_spec.BrowserFFTSpec()
    audio_dataloader.DataLoader.from_folder(spec, folder_path)

  def test_examples_helper(self):
    root = self._get_folder_path('test_examples_helper')
    write_file(root, 'a/1.wav')
    write_file(root, 'a/2.wav')
    write_file(root, 'b/1.wav')
    write_file(root, 'b/README')  # Ignored
    write_file(root, 'a/b/c/d.wav')  # Ignored
    write_file(root, 'AUTHORS.md')  # Ignored
    write_file(root, 'temp.wav')  # Ignored

    def is_wav_files(name):
      return name.endswith('.wav')

    def fullpath(name):
      return os.path.join(root, name)

    helper = audio_dataloader.ExamplesHelper(root, is_wav_files)
    self.assertEqual(helper.sorted_cateogries, ['a', 'b'])
    self.assertEqual(
        helper.examples_and_labels(),
        ([fullpath('a/1.wav'),
          fullpath('a/2.wav'),
          fullpath('b/1.wav')], ['a', 'a', 'b']))
    self.assertEqual(
        helper.examples_and_label_indices(),
        ([fullpath('a/1.wav'),
          fullpath('a/2.wav'),
          fullpath('b/1.wav')], [0, 0, 1]))

  def test_no_audio_files_found(self):
    folder_path = self._get_folder_path('test_no_audio_files_found')
    write_sample(folder_path, 'unknown', '2s.bak', 44100, 2, value=1)
    with self.assertRaisesRegexp(ValueError, 'No audio files found'):
      spec = MockSpec(model_dir=folder_path)
      audio_dataloader.DataLoader.from_folder(spec, folder_path)

  def test_from_folder(self):
    folder_path = self._get_folder_path('test_from_folder')
    write_sample(folder_path, 'background', '2s.wav', 44100, 2, value=0)
    write_sample(folder_path, 'command1', '1s.wav', 44100, 1, value=1)
    # Too short, skipped.
    write_sample(folder_path, 'command1', '0.1s.wav', 44100, .1, value=2)
    # Not long enough for 2 files, the remaining .5s will be skipped.
    write_sample(folder_path, 'command2', '1.5s.wav', 44100, 1.5, value=3)
    # Skipped, too short.
    write_sample(folder_path, 'command0', '0.1s.wav', 4410, .1, value=4)
    # Resampled, after resample, the content becomes [4 5 5 ... 4 5 4]
    write_sample(folder_path, 'command0', '1.8s.wav', 4410, 1.8, value=5)
    # Ignored due to wrong file extension
    write_sample(folder_path, 'command0', '1.8s.bak', 4410, 1.8, value=6)

    spec = MockSpec(model_dir=folder_path)
    loader = audio_dataloader.DataLoader.from_folder(spec, folder_path)

    # 6 files with .wav extennsion
    self.assertEqual(len(loader), 6)
    self.assertEqual(loader.index_to_label,
                     ['background', 'command0', 'command1', 'command2'])

    # 5 valid audio snippets
    self.assertEqual(len(list(loader.gen_dataset())), 5)


if __name__ == '__main__':
  tf.test.main()
