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
"""Tests for metadata_loader."""

import tensorflow as tf
from tensorflow_examples.lite.model_maker.core.data_util import metadata_loader
from tensorflow_examples.lite.model_maker.core import test_util


class MetadataLoaderTest(tf.test.TestCase):

  def test_from_file_name(self):
    loader = metadata_loader.MetadataLoader.from_file_name()
    metadata = loader.load("/tmp/searcher_dataset/burger.jpg")
    self.assertEqual(metadata, "burger")

  def test_from_dat_file(self):

    loader = metadata_loader.MetadataLoader.from_dat_file()

    expected_sparrow_metadata = "{ 'type': 'sparrow', 'class': 'Aves'}"
    sparrow_metadata = loader.load(test_util.get_test_data_path("sparrow.png"))
    self.assertEqual(sparrow_metadata, expected_sparrow_metadata)

    expected_burger_metadata = b"\x00\x11\x22"
    burger_metadata = loader.load(
        test_util.get_test_data_path("cats_and_dogs.jpg"), mode="rb")
    self.assertEqual(burger_metadata, expected_burger_metadata)


if __name__ == "__main__":
  tf.test.main()
