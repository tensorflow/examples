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
import unittest

import tensorflow as tf

from tensorflow_examples.lite.model_maker.core import test_util
from tensorflow_examples.lite.model_maker.demo import audio_classification_demo
from tflite_model_maker import audio_classifier

from_folder_fn = audio_classifier.DataLoader.from_folder


def patch_data_loader():
  """Patch to train partial dataset rather than all of them."""

  def side_effect(*args, **kwargs):
    tf.compat.v1.logging.info('Train on partial dataset')
    # This takes around 8 mins as it caches all files in the folder.
    # We should be able to address this issue once the dataset is lazily loaded.
    data_loader = from_folder_fn(*args, **kwargs)
    if len(data_loader) > 10:  # Trim dataset to at most 10.
      data_loader._size = 10
      # TODO(b/171449557): Change this once the dataset is lazily loaded.
      data_loader._dataset = data_loader._dataset.take(10)
    return data_loader

  return unittest.mock.patch.object(
      audio_classifier.DataLoader, 'from_folder', side_effect=side_effect)


@unittest.skipIf(tf.__version__ < '2.5',
                 'Audio Classification requires TF 2.5 or later')
class AudioClassificationDemoTest(tf.test.TestCase):

  def test_audio_classification_demo(self):
    with patch_data_loader():
      with tempfile.TemporaryDirectory() as temp_dir:
        # Use cached training data if exists.
        data_dir = audio_classification_demo.download_speech_commands_dataset(
            cache_dir=test_util.get_cache_dir(temp_dir,
                                              'mini_speech_commands.zip'),
            file_hash='4b8a67bae2973844e84fa7ac988d1a44')

        tflite_filename = os.path.join(temp_dir, 'model.tflite')
        label_filename = os.path.join(temp_dir, 'labels.txt')
        audio_classification_demo.run(
            'audio_browser_fft',
            data_dir,
            'mini_speech_command',
            temp_dir,
            epochs=1,
            batch_size=1)

        self.assertTrue(tf.io.gfile.exists(tflite_filename))
        self.assertGreater(os.path.getsize(tflite_filename), 0)

        self.assertFalse(tf.io.gfile.exists(label_filename))


if __name__ == '__main__':
  # Load compressed models from tensorflow_hub
  os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'
  tf.test.main()
