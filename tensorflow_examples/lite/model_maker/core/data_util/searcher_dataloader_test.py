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
"""Tests for searcher_dataloader."""

import numpy as np
import tensorflow as tf
from tensorflow_examples.lite.model_maker.core.data_util import searcher_dataloader
from tensorflow_lite_support.python.task.core.proto import base_options_pb2
from tensorflow_lite_support.python.task.vision import image_embedder
from tensorflow_examples.lite.model_maker.core import test_util

_BaseOptions = base_options_pb2.BaseOptions
_ImageEmbedder = image_embedder.ImageEmbedder
_ImageEmbedderOptions = image_embedder.ImageEmbedderOptions
_SearcherDataLoader = searcher_dataloader.DataLoader


class SearcherDataloaderTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.tflite_path = test_util.get_test_data_path(
        "mobilenet_v2_035_96_embedder_with_metadata.tflite")

  def test_concat_dataset(self):
    dataset1 = np.random.rand(2, 1280)
    dataset2 = np.random.rand(2, 1280)
    dataset3 = np.random.rand(1, 1280)
    metadata = ["0", "1", b"\x11\x22", b"\x33\x44", "4"]
    data = _SearcherDataLoader(embedder_path=self.tflite_path)
    data._dataset = dataset1
    data._cache_dataset_list = [dataset2, dataset3]
    data._metadata = metadata

    self.assertTrue((data.dataset == np.vstack([dataset1, dataset2,
                                                dataset3])).all())
    self.assertEqual(data.metadata, metadata)

  def test_append(self):
    dataset1 = np.random.rand(4, 1280)
    metadata1 = ["0", "1", b"\x11\x22", "3"]
    data_loader1 = _SearcherDataLoader(embedder_path=self.tflite_path)
    data_loader1._dataset = dataset1
    data_loader1._metadata = metadata1

    dataset2 = np.random.rand(2, 1280)
    metadata2 = [b"\x33\x44", "5"]
    data_loader2 = _SearcherDataLoader(embedder_path=self.tflite_path)
    data_loader2._dataset = dataset2
    data_loader2._metadata = metadata2

    data_loader1.append(data_loader2)
    self.assertEqual(data_loader1.dataset.shape, (6, 1280))
    self.assertEqual(data_loader1.metadata,
                     ["0", "1", b"\x11\x22", "3", b"\x33\x44", "5"])

  def test_init(self):
    # Initializes from embedder first and then generates the embedding dataset
    # from raw input leverage embedder later.
    data = _SearcherDataLoader(embedder_path=self.tflite_path)
    self.assertEqual(data.dataset.shape, (0,))
    self.assertEqual(data.metadata, [])

    # Initializes directly from the dataset and metadata.
    dataset = np.random.rand(2, 512)
    metadata = ["0", b"\66"]
    data = _SearcherDataLoader(dataset=dataset, metadata=metadata)
    self.assertTrue((data.dataset == dataset).all())
    self.assertEqual(data.metadata, metadata)


if __name__ == "__main__":
  tf.test.main()
