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

import os
import tempfile
from unittest.mock import patch

import tensorflow as tf

from tensorflow_examples.lite.model_maker.core import test_util
from tensorflow_examples.lite.model_maker.core.data_util.image_dataloader import ImageClassifierDataLoader
from tensorflow_examples.lite.model_maker.demo import image_classification_demo


def get_cache_dir():
  return os.path.join(test_util.get_test_data_path('demo'), 'testdata')


from_folder_fn = ImageClassifierDataLoader.from_folder


def patch_data_loader():
  """Patch to train partial dataset rather than all of them."""

  def side_effect(*args, **kwargs):
    tf.compat.v1.logging.info('Train on partial dataset')
    data_loader = from_folder_fn(*args, **kwargs)
    if data_loader.size > 10:  # Trim dataset to at most 10.
      data_loader.size = 10
      data_loader.dataset = data_loader.dataset.take(data_loader.size)
    return data_loader

  return patch.object(
      ImageClassifierDataLoader, 'from_folder', side_effect=side_effect)


class ImageClassificationDemoTest(tf.test.TestCase):

  def test_image_classification_demo(self):
    with patch_data_loader():
      with tempfile.TemporaryDirectory() as temp_dir:
        # Use cached training data if exists.
        data_dir = image_classification_demo.download_demo_data(
            cache_dir=get_cache_dir(),
            file_hash='6f87fb78e9cc9ab41eff2015b380011d')

        tflite_filename = os.path.join(temp_dir, 'model.tflite')
        label_filename = os.path.join(temp_dir, 'labels.txt')
        image_classification_demo.run(
            data_dir,
            temp_dir,
            spec='efficientnet_lite0',
            epochs=1,
            batch_size=1)

        self.assertTrue(tf.io.gfile.exists(tflite_filename))
        self.assertGreater(os.path.getsize(tflite_filename), 0)

        self.assertFalse(tf.io.gfile.exists(label_filename))


if __name__ == '__main__':
  tf.test.main()
