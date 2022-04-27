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
"""Image DataLoader for Searcher task."""

import imghdr
import logging
import os

import numpy as np
from tensorflow_examples.lite.model_maker.core.data_util import metadata_loader
from tensorflow_examples.lite.model_maker.core.data_util import searcher_dataloader
from tensorflow_lite_support.python.task.core.proto import base_options_pb2
from tensorflow_lite_support.python.task.vision import image_embedder
from tensorflow_lite_support.python.task.vision.core import tensor_image

_MetadataType = metadata_loader.MetadataType
_BaseOptions = base_options_pb2.BaseOptions


class DataLoader(searcher_dataloader.DataLoader):
  """DataLoader class for Image Searcher Task."""

  def __init__(
      self,
      embedder: image_embedder.ImageEmbedder,
      metadata_type: _MetadataType = _MetadataType.FROM_FILE_NAME) -> None:
    """Initializes DataLoader for Image Searcher task.

    Args:
      embedder: Embedder to generate embedding from raw input image.
      metadata_type: Type of MetadataLoader to load metadata for each input
        data. By default, load the file name as metadata for each input data.
    """
    self._embedder = embedder
    super().__init__(embedder_path=embedder.options.base_options.file_name)

    # Creates the metadata loader.
    if metadata_type is _MetadataType.FROM_FILE_NAME:
      self._metadata_loader = metadata_loader.MetadataLoader.from_file_name()
    elif metadata_type is _MetadataType.FROM_DAT_FILE:
      self._metadata_loader = metadata_loader.MetadataLoader.from_dat_file()
    else:
      raise ValueError("Unsuported metadata_type.")

  @classmethod
  def create(
      cls,
      image_embedder_path: str,
      metadata_type: _MetadataType = _MetadataType.FROM_FILE_NAME
  ) -> "DataLoader":
    """Creates DataLoader for the Image Searcher task.

    Args:
      image_embedder_path: Path to the ".tflite" image embedder model.
      metadata_type: Type of MetadataLoader to load metadata for each input
        image based on image path. By default, load the file name as metadata
        for each input image.

    Returns:
      DataLoader object created for the Image Searcher task.
    """
    # Creates ImageEmbedder.
    image_embedder_path = os.path.abspath(image_embedder_path)
    base_options = _BaseOptions(file_name=image_embedder_path)
    options = image_embedder.ImageEmbedderOptions(base_options=base_options)
    embedder = image_embedder.ImageEmbedder.create_from_options(options)

    return cls(embedder, metadata_type)

  def load_from_folder(self, path: str) -> None:
    """Loads image data from folder.

    Users can load images from different folders one by one. For instance,
    ```
    # Creates data_loader instance.
    data_loader = image_searcher_dataloader.DataLoader.create(tflite_path)

    # Loads images, first from `image_path1` and secondly from `image_path2`.
    data_loader.load_from_folder(image_path1)
    data_loader.load_from_folder(image_path2)
    ```

    Args:
      path: image directory to be loaded.
    """
    embedding_list = []
    metadata_list = []

    # Gets the image files in the folder and loads images.
    for root, _, files in os.walk(path):
      for name in files:
        image_path = os.path.join(root, name)
        if imghdr.what(image_path) is None:
          continue

        image = tensor_image.TensorImage.create_from_file(image_path)
        try:
          embedding = self._embedder.embed(
              image).embeddings[0].feature_vector.value_float
        except (RuntimeError, ValueError) as e:
          logging.warning("Can't get the embedding of %s with the error %s",
                          image_path, e)
          continue

        embedding_list.append(embedding)
        metadata = self._metadata_loader.load(image_path)
        metadata_list.append(metadata)

    cache_dataset = np.stack(embedding_list)
    self._cache_dataset_list.append(cache_dataset)
    self._metadata = self._metadata + metadata_list
