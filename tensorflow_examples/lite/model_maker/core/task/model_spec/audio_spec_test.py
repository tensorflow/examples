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
import unittest

import numpy as np
from packaging import version
import tensorflow.compat.v2 as tf
from tensorflow_examples.lite.model_maker.core.task import configs
from tensorflow_examples.lite.model_maker.core.task import model_util
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
      np.random.randint(low=0, high=num_classes,
                        size=total_samples).astype('int32'))
  dataset = tf.data.Dataset.zip((wav_ds, labels))
  dataset = spec.preprocess_ds(dataset)

  @tf.function
  def _one_hot_encoding_label(wav, label):
    return wav, tf.one_hot(label, num_classes)

  dataset = dataset.map(_one_hot_encoding_label)

  dataset = dataset.batch(batch_size)

  return dataset


class BaseSpecTest(tf.test.TestCase):

  def testEnsureVersion(self):
    valid_versions = ['2.5.0', '2.5.0rc1', '2.6']
    invalid_versions = [
        '2.4.1',
    ]
    specs = [audio_spec.YAMNetSpec, audio_spec.BrowserFFTSpec]

    tmp_version_fn = audio_spec._get_tf_version
    for spec in specs:
      for valid_version in valid_versions:
        audio_spec._get_tf_version = lambda: valid_version  # pylint: disable=cell-var-from-loop
        spec()

      for valid_version in invalid_versions:
        audio_spec._get_tf_version = lambda: valid_version  # pylint: disable=cell-var-from-loop
        with self.assertRaisesRegex(RuntimeError, '2.5.0'):
          spec()

    audio_spec._get_tf_version = tmp_version_fn


class BaseTest(tf.test.TestCase):

  def _train_and_export(self,
                        spec,
                        num_classes,
                        filename,
                        expected_model_size,
                        quantization_config=None,
                        training=True):
    dataset = _gen_dataset(
        spec, total_samples=10, num_classes=num_classes, batch_size=2, seed=100)
    model = spec.create_model(num_classes)
    if training:
      spec.run_classifier(model, epochs=1, train_ds=dataset, validation_ds=None)

    tflite_filepath = os.path.join(self.get_temp_dir(), filename)
    spec.export_tflite(
        model,
        tflite_filepath,
        index_to_label=['label_{}'.format(i) for i in range(num_classes)],
        quantization_config=quantization_config)

    self.assertNear(
        os.path.getsize(tflite_filepath), expected_model_size, 1000 * 1000)

    return tflite_filepath


@unittest.skipIf(
    version.parse(tf.__version__) < version.parse('2.5'),
    'Audio Classification requires TF 2.5 or later')
class YAMNetSpecTest(BaseTest):

  def _test_preprocess(self, input_shape, input_count, output_shape,
                       output_count):
    spec = audio_spec.YAMNetSpec()
    wav_ds = tf.data.Dataset.from_tensor_slices([tf.ones(input_shape)] *
                                                input_count)
    label_ds = tf.data.Dataset.range(input_count).map(
        lambda x: tf.cast(x, tf.int32))

    ds = tf.data.Dataset.zip((wav_ds, label_ds))
    ds = spec.preprocess_ds(ds)

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
    # No padding on the input.
    self._test_preprocess(
        input_shape=(10,), input_count=2, output_shape=(1024,), output_count=0)
    # Split the input data into trunks
    self._test_preprocess(
        input_shape=(16000 * 2,),
        input_count=2,
        output_shape=(1024,),
        output_count=6)
    self._test_preprocess(
        input_shape=(15600,),
        input_count=1,
        output_shape=(1024,),
        output_count=1)

  def test_create_model(self):
    # Make sure that there is no naming conflicts in the graph.
    spec = audio_spec.YAMNetSpec()
    model = spec.create_model(10)
    model = spec.create_model(10)
    model = spec.create_model(10)
    self.assertEqual(model.input_shape, (None, 1024))
    self.assertEqual(model.output_shape, (None, 10))

  def test_yamnet_two_heads(self):
    tflite_path = self._train_and_export(
        audio_spec.YAMNetSpec(keep_yamnet_and_custom_heads=True),
        num_classes=2,
        filename='two_heads.tflite',
        expected_model_size=15 * 1000 * 1000)
    self.assertEqual(
        2, len(model_util.get_lite_runner(tflite_path).output_details))
    self.assertAllEqual(
        [1, 521],
        model_util.get_lite_runner(tflite_path).output_details[0]['shape'])
    self.assertAllEqual(
        [1, 2],
        model_util.get_lite_runner(tflite_path).output_details[1]['shape'])
    self.assertEqual(
        model_util.extract_tflite_metadata_json(tflite_path), """{
  "name": "yamnet/classification",
  "description": "Recognizes sound events",
  "version": "v1",
  "subgraph_metadata": [
    {
      "input_tensor_metadata": [
        {
          "name": "audio_clip",
          "description": "Input audio clip to be classified.",
          "content": {
            "content_properties_type": "AudioProperties",
            "content_properties": {
              "sample_rate": 16000,
              "channels": 1
            }
          },
          "stats": {
          }
        }
      ],
      "output_tensor_metadata": [
        {
          "name": "yamnet",
          "description": "Scores in range 0..1.0 for each of the 521 output classes.",
          "content": {
            "content_properties_type": "FeatureProperties",
            "content_properties": {
            }
          },
          "stats": {
            "max": [
              1.0
            ],
            "min": [
              0.0
            ]
          },
          "associated_files": [
            {
              "name": "yamnet_labels.txt",
              "description": "Labels for categories that the model can recognize.",
              "type": "TENSOR_AXIS_LABELS"
            }
          ]
        },
        {
          "name": "custom",
          "description": "Scores in range 0..1.0 for each output classes.",
          "content": {
            "content_properties_type": "FeatureProperties",
            "content_properties": {
            }
          },
          "stats": {
            "max": [
              1.0
            ],
            "min": [
              0.0
            ]
          },
          "associated_files": [
            {
              "name": "custom_labels.txt",
              "description": "Labels for categories that the model can recognize.",
              "type": "TENSOR_AXIS_LABELS"
            }
          ]
        }
      ]
    }
  ],
  "author": "TensorFlow Lite Model Maker",
  "license": "Apache License. Version 2.0 http://www.apache.org/licenses/LICENSE-2.0.",
  "min_parser_version": "1.3.0"
}
""")

  def test_yamnet_single_head(self):
    tflite_path = self._train_and_export(
        audio_spec.YAMNetSpec(keep_yamnet_and_custom_heads=False),
        num_classes=2,
        filename='single_head.tflite',
        expected_model_size=13 * 1000 * 1000)
    self.assertEqual(
        1, len(model_util.get_lite_runner(tflite_path).output_details))
    self.assertAllEqual(
        [1, 2],
        model_util.get_lite_runner(tflite_path).output_details[0]['shape'])
    self.assertEqual(
        model_util.extract_tflite_metadata_json(tflite_path), """{
  "name": "yamnet/classification",
  "description": "Recognizes sound events",
  "version": "v1",
  "subgraph_metadata": [
    {
      "input_tensor_metadata": [
        {
          "name": "audio_clip",
          "description": "Input audio clip to be classified.",
          "content": {
            "content_properties_type": "AudioProperties",
            "content_properties": {
              "sample_rate": 16000,
              "channels": 1
            }
          },
          "stats": {
          }
        }
      ],
      "output_tensor_metadata": [
        {
          "name": "custom",
          "description": "Scores in range 0..1.0 for each output classes.",
          "content": {
            "content_properties_type": "FeatureProperties",
            "content_properties": {
            }
          },
          "stats": {
            "max": [
              1.0
            ],
            "min": [
              0.0
            ]
          },
          "associated_files": [
            {
              "name": "custom_labels.txt",
              "description": "Labels for categories that the model can recognize.",
              "type": "TENSOR_AXIS_LABELS"
            }
          ]
        }
      ]
    }
  ],
  "author": "TensorFlow Lite Model Maker",
  "license": "Apache License. Version 2.0 http://www.apache.org/licenses/LICENSE-2.0.",
  "min_parser_version": "1.3.0"
}
""")

  def test_no_metadata(self):
    audio_spec.ENABLE_METADATA = False
    tflite_path = self._train_and_export(
        audio_spec.YAMNetSpec(keep_yamnet_and_custom_heads=True),
        num_classes=2,
        filename='two_heads.tflite',
        expected_model_size=15 * 1000 * 1000)
    self.assertEqual(
        2, len(model_util.get_lite_runner(tflite_path).output_details))
    with self.assertRaisesRegex(ValueError, 'The model does not have metadata'):
      model_util.extract_tflite_metadata_json(tflite_path)
    audio_spec.ENABLE_METADATA = True

  def test_binary_classification(self):
    self._train_and_export(
        audio_spec.YAMNetSpec(keep_yamnet_and_custom_heads=True),
        num_classes=2,
        filename='binary_classification.tflite',
        expected_model_size=15 * 1000 * 1000)

  def test_dynamic_range_quantization(self):
    self._train_and_export(
        audio_spec.YAMNetSpec(keep_yamnet_and_custom_heads=True),
        num_classes=5,
        filename='basic_5_classes_training.tflite',
        expected_model_size=4 * 1000 * 1000,
        quantization_config=configs.QuantizationConfig.for_dynamic())


@unittest.skipIf(
    version.parse(tf.__version__) < version.parse('2.5'),
    'Audio Classification requires TF 2.5 or later')
class BrowserFFTSpecTest(BaseTest):

  @classmethod
  def setUpClass(cls):
    super(BrowserFFTSpecTest, cls).setUpClass()
    cls._spec = audio_spec.BrowserFFTSpec()

  def test_model_initialization(self):
    model = self._spec.create_model(10)

    self.assertEqual(self._spec._preprocess_model.input_shape,
                     (None, self._spec.EXPECTED_WAVEFORM_LENGTH))
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

  def test_dynamic_range_quantization(self):
    self._train_and_export(
        audio_spec.BrowserFFTSpec(),
        num_classes=2,
        filename='binary_classification.tflite',
        expected_model_size=1 * 1000 * 1000,
        quantization_config=configs.QuantizationConfig.for_dynamic(),
        training=False)  # Training results Nan values with the current scheme.

  def test_binary_classification(self):
    self._train_and_export(
        audio_spec.BrowserFFTSpec(),
        num_classes=2,
        filename='binary_classification.tflite',
        expected_model_size=6 * 1000 * 1000)

  def test_basic_training(self):
    tflite_path = self._train_and_export(
        audio_spec.BrowserFFTSpec(),
        num_classes=5,
        filename='basic_5_classes_training.tflite',
        expected_model_size=6 * 1000 * 1000)
    self.assertEqual(
        model_util.extract_tflite_metadata_json(tflite_path), """{
  "name": "AudioClassifier",
  "description": "Identify the most prominent type in the audio clip from a known set of categories.",
  "version": "v1",
  "subgraph_metadata": [
    {
      "input_tensor_metadata": [
        {
          "name": "audio_clip",
          "description": "Input audio clip to be classified.",
          "content": {
            "content_properties_type": "AudioProperties",
            "content_properties": {
              "sample_rate": 44100,
              "channels": 1
            }
          },
          "stats": {
          }
        }
      ],
      "output_tensor_metadata": [
        {
          "name": "probability",
          "description": "Scores of the labels respectively.",
          "content": {
            "content_properties_type": "FeatureProperties",
            "content_properties": {
            }
          },
          "stats": {
            "max": [
              1.0
            ],
            "min": [
              0.0
            ]
          },
          "associated_files": [
            {
              "name": "probability_labels.txt",
              "description": "Labels for categories that the model can recognize.",
              "type": "TENSOR_AXIS_LABELS"
            }
          ]
        }
      ]
    }
  ],
  "author": "TensorFlow Lite Model Maker",
  "license": "Apache License. Version 2.0 http://www.apache.org/licenses/LICENSE-2.0.",
  "min_parser_version": "1.3.0"
}
""")


if __name__ == '__main__':
  # Load compressed models from tensorflow_hub
  os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'
  tf.test.main()
