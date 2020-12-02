# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

import inspect
import os

import tensorflow.compat.v2 as tf
from tensorflow_examples.lite.model_maker.core import test_util
from tensorflow_examples.lite.model_maker.core.export_format import ExportFormat
from tensorflow_examples.lite.model_maker.core.task import custom_model


class MockCustomModel(custom_model.CustomModel):

  DEFAULT_EXPORT_FORMAT = (ExportFormat.TFLITE, ExportFormat.LABEL)
  ALLOWED_EXPORT_FORMAT = (ExportFormat.TFLITE, ExportFormat.LABEL,
                           ExportFormat.SAVED_MODEL, ExportFormat.TFJS)

  def _export_labels(self, label_filepath):
    with open(label_filepath, 'w') as f:
      f.write('0\n')

  def train(self, train_data, validation_data=None, **kwargs):
    pass

  def evaluate(self, data, **kwargs):
    pass


class CustomModelTest(tf.test.TestCase):

  def setUp(self):
    super(CustomModelTest, self).setUp()
    self.model = MockCustomModel(
        model_spec=None,
        shuffle=False)
    self.model.model = test_util.build_model(input_shape=[4], num_classes=2)

  def _check_nonempty_dir(self, dirpath):
    self.assertTrue(os.path.isdir(dirpath))
    self.assertNotEmpty(os.listdir(dirpath))

  def _check_nonempty_file(self, filepath):
    self.assertTrue(os.path.isfile(filepath))
    self.assertGreater(os.path.getsize(filepath), 0)

  def test_export_saved_model(self):
    saved_model_filepath = os.path.join(self.get_temp_dir(), 'saved_model/')
    self.model._export_saved_model(saved_model_filepath)
    self._check_nonempty_dir(saved_model_filepath)

  def test_export(self):
    # Test whether there's naming conflict in different export functions.
    params1 = inspect.signature(
        self.model._export_saved_model).parameters.keys()
    params2 = inspect.signature(self.model._export_tflite).parameters.keys()
    self.assertTrue(params1.isdisjoint(params2))

    export_path = os.path.join(self.get_temp_dir(), 'export0/')
    self.model.export(export_path)
    self._check_nonempty_file(os.path.join(export_path, 'model.tflite'))
    self.assertFalse(os.path.exists(os.path.join(export_path, 'labels.txt')))

    export_path = os.path.join(self.get_temp_dir(), 'export1/')
    self.model.export(export_path, with_metadata=False)
    self._check_nonempty_file(os.path.join(export_path, 'model.tflite'))
    self._check_nonempty_file(os.path.join(export_path, 'labels.txt'))

    export_path = os.path.join(self.get_temp_dir(), 'export2/')
    self.model.export(
        export_path,
        export_format=[
            ExportFormat.TFLITE, ExportFormat.LABEL, ExportFormat.SAVED_MODEL
        ],
        with_metadata=True)
    self._check_nonempty_file(os.path.join(export_path, 'model.tflite'))
    self._check_nonempty_file(os.path.join(export_path, 'labels.txt'))
    self._check_nonempty_dir(os.path.join(export_path, 'saved_model'))

    export_path = os.path.join(self.get_temp_dir(), 'export3/')
    self.model.export(
        export_path,
        export_format=[ExportFormat.TFLITE, ExportFormat.SAVED_MODEL],
        with_metadata=True,
        include_optimizer=False)
    self._check_nonempty_file(os.path.join(export_path, 'model.tflite'))
    self._check_nonempty_dir(os.path.join(export_path, 'saved_model'))

    export_path = os.path.join(self.get_temp_dir(), 'export4/')
    self.model.export(export_path, export_format=[ExportFormat.TFJS])
    expected_file = os.path.join(export_path, 'tfjs', 'model.json')
    self._check_nonempty_file(expected_file)


if __name__ == '__main__':
  tf.test.main()
