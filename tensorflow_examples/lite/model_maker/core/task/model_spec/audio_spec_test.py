# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for audio specs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow_examples.lite.model_maker.core.task.model_spec import audio_spec


def _gen_dataset(spec, total_samples, num_classes, batch_size, seed):

  def fill_shape(new_shape):

    @tf.function
    def fn(value):
      return tf.cast(tf.fill(dims=new_shape, value=value), tf.float32)

    return fn

  wav_ds = tf.data.experimental.RandomDataset(seed=seed).take(total_samples)
  wav_ds = wav_ds.map(fill_shape([
      spec.target_sample_rate,
  ]))

  labels = tf.data.Dataset.from_tensor_slices(
      np.random.randint(low=0, high=num_classes, size=total_samples))
  dataset = tf.data.Dataset.zip((wav_ds, labels))
  dataset = spec.preprocess_ds(dataset)

  @tf.function
  def _one_hot_encoding_label(wav, label):
    return wav, tf.one_hot(label, num_classes)

  dataset = dataset.map(_one_hot_encoding_label)

  dataset = dataset.batch(batch_size)

  return dataset


class YAMNetSpecTest(tf.test.TestCase):

  @classmethod
  def setUpClass(cls):
    super(YAMNetSpecTest, cls).setUpClass()
    cls._spec = audio_spec.YAMNetSpec()

  def _test_preprocess(self, input_shape, input_count, output_shape,
                       output_count):
    wav_ds = tf.data.Dataset.from_tensor_slices([tf.ones(input_shape)] *
                                                input_count)
    label_ds = tf.data.Dataset.range(input_count)

    ds = tf.data.Dataset.zip((wav_ds, label_ds))
    ds = self._spec.preprocess_ds(ds)

    chunks = output_count // input_count

    cnt = 0
    for item, label in ds:
      cnt += 1
    self.assertEqual(cnt, output_count)

    # More thorough checks.
    cnt = 0
    for item, label in ds:
      self.assertEqual(output_shape, item.shape)
      self.assertEqual(label, cnt // chunks)
      cnt += 1

  def test_preprocess(self):
    # YAMNet does padding on the input.
    self._test_preprocess(
        input_shape=(10,), input_count=2, output_shape=(1024,), output_count=2)
    # Split the input data into trunks
    self._test_preprocess(
        input_shape=(16000 * 2,),
        input_count=2,
        output_shape=(1024,),
        output_count=8)
    self._test_preprocess(
        input_shape=(8000,),
        input_count=1,
        output_shape=(1024,),
        output_count=1)

  def test_create_model(self):
    # Make sure that there is no naming conflicts.
    model = self._spec.create_model(10)
    model = self._spec.create_model(10)
    model = self._spec.create_model(10)
    self.assertEqual(model.input_shape, (None, 1024))
    self.assertEqual(model.output_shape, (None, 10))

  def _train(self, total_samples, num_classes, batch_size, seed):
    tf.keras.backend.clear_session()

    tf.random.set_seed(seed)
    np.random.seed(seed)

    dataset = _gen_dataset(self._spec, total_samples, num_classes, batch_size,
                           seed)
    model = self._spec.create_model(num_classes)
    self._spec.run_classifier(
        model, epochs=1, train_ds=dataset, validation_ds=dataset)

    # Test tflite export
    tflite_filepath = os.path.join(self.get_temp_dir(), 'model.tflite')
    self._spec.export_tflite(model, tflite_filepath)
    expected_model_size = 13 * 1000 * 1000
    self.assertNear(
        os.path.getsize(tflite_filepath), expected_model_size, 1000 * 1000)

  def test_binary_classification(self):
    self._train(total_samples=10, num_classes=2, batch_size=2, seed=100)

  def test_basic_training(self):
    self._train(total_samples=20, num_classes=3, batch_size=2, seed=100)


class BrowserFFTSpecTest(tf.test.TestCase):

  @classmethod
  def setUpClass(cls):
    super(BrowserFFTSpecTest, cls).setUpClass()
    cls._spec = audio_spec.BrowserFFTSpec()

  def test_model_initialization(self):
    model = self._spec.create_model(10)

    self.assertEqual(self._spec._preprocess_model.input_shape,
                     (None, self._spec.expected_waveform_len))
    self.assertEqual(self._spec._preprocess_model.output_shape,
                     (None, None, 232, 1))
    self.assertEqual(self._spec._tfjs_sc_model.input_shape, (None, 43, 232, 1))
    self.assertEqual(self._spec._tfjs_sc_model.output_shape, (None, 20))
    self.assertEqual(model.input_shape, (None, 43, 232, 1))
    self.assertEqual(model.output_shape, (None, 10))

  def test_create_model(self):
    # Make sure that there is no naming conflicts.
    self._spec.create_model(100)
    self._spec.create_model(100)
    self._spec.create_model(100)

    tf.keras.backend.clear_session()
    # Binary classification is not supported yet.
    with self.assertRaises(ValueError):
      self._spec.create_model(0)
    tf.keras.backend.clear_session()
    with self.assertRaises(ValueError):
      self._spec.create_model(1)
    tf.keras.backend.clear_session()
    # It's more efficient to use BinaryClassification when num_classes=2, but
    # this is still supported (slightly less efficient).
    self._spec.create_model(20)
    tf.keras.backend.clear_session()

  def _train(self, total_samples, num_classes, batch_size, seed):
    tf.keras.backend.clear_session()

    tf.random.set_seed(seed)
    np.random.seed(seed)

    dataset = _gen_dataset(self._spec, total_samples, num_classes, batch_size,
                           seed)
    model = self._spec.create_model(num_classes)
    self._spec.run_classifier(
        model, epochs=1, train_ds=dataset, validation_ds=dataset)

    # Test tflite export
    tflite_filepath = os.path.join(self.get_temp_dir(), 'model.tflite')
    self._spec.export_tflite(model, tflite_filepath)
    expected_model_size = 6 * 1000 * 1000
    self.assertNear(
        os.path.getsize(tflite_filepath), expected_model_size, 1000 * 1000)

  def test_binary_classification(self):
    self._train(total_samples=10, num_classes=2, batch_size=2, seed=100)

  def test_basic_training(self):
    self._train(total_samples=20, num_classes=3, batch_size=2, seed=100)


if __name__ == '__main__':
  # Load compressed models from tensorflow_hub
  os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'
  tf.test.main()
