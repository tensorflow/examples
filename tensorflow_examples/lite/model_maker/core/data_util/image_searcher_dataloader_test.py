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

from absl.testing import parameterized

import tensorflow as tf
from tensorflow_examples.lite.model_maker.core.data_util import image_searcher_dataloader
from tensorflow_examples.lite.model_maker.core.data_util import metadata_loader
from tensorflow_examples.lite.model_maker.core import test_util


class ImageSearcherDataloaderTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.tflite_path = test_util.get_test_data_path(
        "mobilenet_v2_035_96_embedder_with_metadata.tflite")
    self.image_dir1 = test_util.get_test_data_path("food")
    self.image_dir2 = test_util.get_test_data_path("animals")

  @parameterized.parameters(
      (False, 1.37590),
      (True, 0.0494329),
  )
  def test_from_folder(self, l2_normalize, expected_value):
    data_loader = image_searcher_dataloader.DataLoader.create(
        self.tflite_path, l2_normalize=l2_normalize)

    data_loader.load_from_folder(self.image_dir1)
    data_loader.load_from_folder(self.image_dir2)
    self.assertLen(data_loader, 3)
    self.assertEqual(data_loader.dataset.shape, (3, 1280))
    self.assertAlmostEqual(data_loader.dataset[0][0], expected_value, places=6)
    # The order of file may be different.
    self.assertEqual(set(data_loader.metadata),
                     set(["burger", "sparrow", "cats_and_dogs"]))

  def test_from_folder_binary_metadata(self,):
    image_dir = test_util.get_test_data_path("images_with_binary_metadata")
    data_loader = image_searcher_dataloader.DataLoader.create(
        self.tflite_path,
        metadata_type=metadata_loader.MetadataType.FROM_DAT_FILE)

    # Loads from binary metadata.
    data_loader.load_from_folder(image_dir, mode="rb")
    self.assertLen(data_loader, 2)
    self.assertEqual(data_loader.dataset.shape, (2, 1280))
    # The order of file may be different.
    self.assertEqual(set(data_loader.metadata), set([b"\x11\x33", b"\x00\x44"]))


if __name__ == "__main__":
  tf.test.main()
