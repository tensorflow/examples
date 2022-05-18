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
"""Tests for text_searcher_dataloader."""

from absl.testing import parameterized

import tensorflow as tf
from tensorflow_examples.lite.model_maker.core.data_util import text_searcher_dataloader
from tensorflow_examples.lite.model_maker.core import test_util


class TextSearcherDataloaderTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.tflite_path = test_util.get_test_data_path(
        "regex_one_embedding_with_metadata.tflite")
    self.text_csv_file1 = test_util.get_test_data_path("movies.csv")
    self.text_csv_file2 = test_util.get_test_data_path("trips.csv")

  @parameterized.parameters(
      (False, 0.0301549),
      (True, 0.538059),
  )
  def test_from_csv(self, l2_normalize, expected_value):
    data_loader = text_searcher_dataloader.DataLoader.create(
        self.tflite_path, l2_normalize=l2_normalize)
    data_loader.load_from_csv(
        self.text_csv_file1, text_column="text", metadata_column="metadata")
    data_loader.load_from_csv(
        self.text_csv_file2, text_column="text", metadata_column="metadata")
    self.assertLen(data_loader, 3)
    self.assertEqual(data_loader.dataset.shape, (3, 16))
    self.assertAlmostEqual(data_loader.dataset[0][0], expected_value, places=6)
    self.assertEqual(data_loader.metadata,
                     ["good movie", "charming trip", "great trip"])


if __name__ == "__main__":
  tf.test.main()
