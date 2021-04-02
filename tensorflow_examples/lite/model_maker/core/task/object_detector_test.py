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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import filecmp
import os

from absl import logging
import tensorflow.compat.v2 as tf
from tensorflow_examples.lite.model_maker.core import compat
from tensorflow_examples.lite.model_maker.core import test_util
from tensorflow_examples.lite.model_maker.core.data_util import object_detector_dataloader
from tensorflow_examples.lite.model_maker.core.export_format import ExportFormat
from tensorflow_examples.lite.model_maker.core.export_format import QuantizationType
from tensorflow_examples.lite.model_maker.core.task import model_spec
from tensorflow_examples.lite.model_maker.core.task import object_detector


class ObjectDetectorTest(tf.test.TestCase):

  def testEfficientDetLite0(self):
    # Gets model specification.
    spec = model_spec.get('efficientdet_lite0')

    # Prepare data.
    images_dir, annotations_dir, label_map = test_util.create_pascal_voc(
        self.get_temp_dir())
    data = object_detector_dataloader.DataLoader.from_pascal_voc(
        images_dir, annotations_dir, label_map)

    # Train the model.
    task = object_detector.create(data, spec, batch_size=1, epochs=1)
    self.assertEqual(spec.config.num_classes, 2)

    # Evaluate trained model
    metrics = task.evaluate(data)
    self.assertIsInstance(metrics, dict)
    self.assertGreaterEqual(metrics['AP'], 0)

    # Export the model to saved model.
    output_path = os.path.join(self.get_temp_dir(), 'saved_model')
    task.export(self.get_temp_dir(), export_format=ExportFormat.SAVED_MODEL)
    self.assertTrue(os.path.isdir(output_path))
    self.assertNotEqual(len(os.listdir(output_path)), 0)

    # Export the model to TFLite model.
    output_path = os.path.join(self.get_temp_dir(), 'float.tflite')
    task.export(
        self.get_temp_dir(),
        tflite_filename='float.tflite',
        quantization_type=QuantizationType.FP32,
        export_format=ExportFormat.TFLITE,
        with_metadata=True,
        export_metadata_json_file=True)
    # Checks the sizes of the float32 TFLite model files in bytes.
    model_size = 13476379
    self.assertNear(os.path.getsize(output_path), model_size, 50000)

    json_output_file = os.path.join(self.get_temp_dir(), 'float.json')
    self.assertTrue(os.path.isfile(json_output_file))
    self.assertGreater(os.path.getsize(json_output_file), 0)
    expected_json_file = test_util.get_test_data_path(
        'efficientdet_lite0_metadata.json')
    self.assertTrue(filecmp.cmp(json_output_file, expected_json_file))

    # Evaluate the TFLite model.
    task.evaluate_tflite(output_path, data)
    self.assertIsInstance(metrics, dict)
    self.assertGreaterEqual(metrics['AP'], 0)

    # Export the model to quantized TFLite model.
    # TODO(b/175173304): Skips the test for stable tensorflow 2.4 for now since
    # it fails. Will revert this change after TF upgrade.
    if tf.__version__.startswith('2.4'):
      return

    # Not include QuantizationType.FP32 here since we have already tested it
    # above together with metadata file test.
    types = (QuantizationType.INT8, QuantizationType.FP16,
             QuantizationType.DYNAMIC)
    # The sizes of the TFLite model files in bytes.
    model_sizes = (4439987, 6840331, 4289875)
    for quantization_type, model_size in zip(types, model_sizes):
      filename = quantization_type.name.lower() + '.tflite'
      output_path = os.path.join(self.get_temp_dir(), filename)
      task.export(
          self.get_temp_dir(),
          quantization_type=quantization_type,
          tflite_filename=filename,
          export_format=ExportFormat.TFLITE)
      self.assertNear(os.path.getsize(output_path), model_size, 50000)


if __name__ == '__main__':
  logging.set_verbosity(logging.ERROR)
  # Load compressed models from tensorflow_hub
  os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'
  compat.setup_tf_behavior(tf_version=2)
  tf.test.main()
