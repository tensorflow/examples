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
"""Recommendation demo test."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import tempfile
from unittest import mock
import zipfile

import tensorflow as tf

from tensorflow_examples.lite.model_maker.core import test_util
from tensorflow_examples.lite.model_maker.core.data_util import recommendation_testutil as _rt
from tensorflow_examples.lite.model_maker.demo import recommendation_demo
from tflite_model_maker import recommendation


def setup_testdata(instance):
  """Setup testdata under download_dir, and unzip data to dataset_dir."""
  if not hasattr(instance, 'test_tempdir'):
    instance.test_tempdir = tempfile.mkdtemp()
  instance.download_dir = os.path.join(instance.test_tempdir, 'download')

  # Copy zip file and unzip.
  os.makedirs(instance.download_dir, exist_ok=True)
  # Use existing copy of data, if exists; otherwise, download it.
  try:
    path = test_util.get_test_data_path('recommendation_movielens')
    zip_file = os.path.join(path, 'ml-1m.zip')
    shutil.copy(zip_file, instance.download_dir)
    with zipfile.ZipFile(zip_file, 'r') as zfile:
      zfile.extractall(instance.download_dir)  # Will generate at 'ml-1m'.
    instance.dataset_dir = os.path.join(instance.download_dir, 'ml-1m')
  except ValueError:
    instance.dataset_dir = recommendation_demo.download_data(
        instance.download_dir)


def patch_data_loader():
  """Patch to train/eval partial dataset rather than all of them."""

  def mocked_init(self, dataset, size, vocab):
    """Mocked init function with a smaller dataset."""
    size = 16  # small size for dataset.
    self._dataset = dataset.take(size)
    self._size = size
    self.vocab = vocab
    self.max_vocab_id = max(self.vocab.keys())

  return mock.patch.object(recommendation.DataLoader, '__init__', mocked_init)


class RecommendationDemoTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    setup_testdata(self)

  def test_recommendation_demo(self):
    with _rt.patch_download_and_extract_data(self.dataset_dir):
      data_dir = recommendation_demo.download_data(self.download_dir)
    self.assertEqual(data_dir, self.dataset_dir)

    export_dir = os.path.join(self.test_tempdir, 'export')
    tflite_filename = os.path.join(export_dir, 'model.tflite')
    with patch_data_loader():
      recommendation_demo.run(
          data_dir,
          export_dir,
          spec='recommendation_bow',
          epochs=1)

    self.assertTrue(tf.io.gfile.exists(tflite_filename))
    self.assertGreater(os.path.getsize(tflite_filename), 0)


if __name__ == '__main__':
  # Load compressed models from tensorflow_hub
  os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'
  tf.test.main()
