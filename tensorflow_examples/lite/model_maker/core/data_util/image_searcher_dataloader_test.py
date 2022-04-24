# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for image_searcher_dataloader."""

import tensorflow as tf
from tensorflow_examples.lite.model_maker.core.data_util import image_searcher_dataloader
from tensorflow_examples.lite.model_maker.core import test_util


class ImageSearcherDataloaderTest(tf.test.TestCase):

  def test_from_folder(self):
    tflite_path = test_util.get_test_data_path(
        "mobilenet_v2_035_96_embedder_with_metadata.tflite")
    image_dir1 = test_util.get_test_data_path("food")
    image_dir2 = test_util.get_test_data_path("animals")

    data_loader = image_searcher_dataloader.DataLoader.create(tflite_path)
    data_loader.load_from_folder(image_dir1)
    data_loader.load_from_folder(image_dir2)
    self.assertLen(data_loader, 3)
    self.assertEqual(data_loader.dataset.shape, (3, 1280))
    self.assertEqual(data_loader.metadata,
                     ["burger", "sparrow", "cats_and_dogs"])


if __name__ == "__main__":
  tf.test.main()
