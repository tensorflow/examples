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
from tensorflow_examples.lite.model_maker.core.data_util.text_dataloader import TextClassifierDataLoader
from tensorflow_examples.lite.model_maker.demo import text_classification_demo


from_csv_fn = TextClassifierDataLoader.from_csv


def patch_data_loader():
  """Patch to train partial dataset rather than all of them."""

  def side_effect(*args, **kwargs):
    tf.compat.v1.logging.info('Train on partial dataset')
    data_loader = from_csv_fn(*args, **kwargs)
    if len(data_loader) > 2:  # Trim dataset to at most 2.
      data_loader._size = 2
      # TODO(b/171449557): Change this once dataset is lazily loaded.
      data_loader._dataset = data_loader._dataset.take(2)
    return data_loader

  return patch.object(
      TextClassifierDataLoader, 'from_csv', side_effect=side_effect)


class TextClassificationDemoTest(tf.test.TestCase):

  def test_text_classification_demo(self):
    with patch_data_loader():
      with tempfile.TemporaryDirectory() as temp_dir:
        # Use cached training data if exists.
        data_dir = text_classification_demo.download_demo_data(
            cache_dir=test_util.get_cache_dir(temp_dir, 'SST-2.zip'),
            file_hash='9f81648d4199384278b86e315dac217c')

        tflite_filename = os.path.join(temp_dir, 'model.tflite')
        label_filename = os.path.join(temp_dir, 'labels.txt')
        vocab_filename = os.path.join(temp_dir, 'vocab')
        # TODO(b/150597348): Bert model is out of memory when export to tflite.
        # Changed to a smaller bert models like mobilebert later for unittest.
        text_classification_demo.run(
            data_dir, temp_dir, spec='average_word_vec', epochs=1, batch_size=1)

        self.assertTrue(tf.io.gfile.exists(tflite_filename))
        self.assertGreater(os.path.getsize(tflite_filename), 0)

        self.assertFalse(tf.io.gfile.exists(label_filename))
        self.assertFalse(tf.io.gfile.exists(vocab_filename))


if __name__ == '__main__':
  # Load compressed models from tensorflow_hub
  os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'
  tf.test.main()
