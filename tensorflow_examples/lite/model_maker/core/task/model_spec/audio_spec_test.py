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

import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow_examples.lite.model_maker.core.task.model_spec import audio_spec


class BaseSpecTest(tf.test.TestCase):

  def test_unable_to_instantiate_baseclass(self):
    with self.assertRaisesRegex(TypeError, 'Can\'t instantiate abstract class'):
      audio_spec.BaseSpec()


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

    def fill_shape(new_shape):

      def fn(value):
        return tf.fill(dims=new_shape, value=value)

      return fn

    tf.random.set_seed(seed)
    np.random.seed(seed)

    wav_ds = tf.data.experimental.RandomDataset(seed=seed).take(total_samples)
    wav_ds = wav_ds.map(fill_shape([
        self._spec.expected_waveform_len,
    ]))

    labels = tf.data.Dataset.from_tensor_slices(
        np.random.randint(low=0, high=num_classes, size=total_samples))
    dataset = tf.data.Dataset.zip((wav_ds, labels))
    dataset = self._spec.preprocess_ds(dataset)
    dataset = dataset.batch(batch_size)

    model = self._spec.create_model(num_classes)
    self._spec.run_classifier(
        model, epochs=1, train_ds=dataset, validation_ds=dataset)

  def test_binary_classification(self):
    self._train(total_samples=10, num_classes=2, batch_size=2, seed=100)

  def test_basic_training(self):
    self._train(total_samples=20, num_classes=3, batch_size=2, seed=100)


if __name__ == '__main__':
  tf.test.main()
