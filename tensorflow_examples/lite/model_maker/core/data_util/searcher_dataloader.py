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
"""DataLoader for Searcher task."""

from typing import AnyStr, List, Optional

import numpy as np
from tensorflow_examples.lite.model_maker.core.api.api_util import mm_export


@mm_export("searcher.DataLoader")
class DataLoader(object):
  """Base DataLoader class for Searcher task."""

  def __init__(
      self,
      embedder_path: Optional[str] = None,
      dataset: Optional[np.ndarray] = None,
      metadata: Optional[List[AnyStr]] = None,
  ) -> None:
    """Initializes DataLoader for Searcher task.

    Args:
      embedder_path: Path to the TFLite Embedder model file.
      dataset: Embedding dataset used to build on-device ScaNN index file. The
        dataset shape should be (dataset_size, embedding_dim). If None,
        `dataset` will be generated from raw input data later.
      metadata:  The metadata for each data in the dataset. The length of
        `metadata` should be same as `dataset` and passed in the same order as
        `dataset`. If `dataset` is set, `metadata` should be set as well.
    """
    self._embedder_path = embedder_path

    # Cache dataset list which can be concatenated to the single dataset. This
    # is used since users may load the data several times, if each time the data
    # are directly concatenated together, memory copy will be costed each time.
    self._cache_dataset_list = []

    if dataset is None:
      # Sets dataset and metadata as empty. Will load them from raw input data
      # later.
      self._dataset = np.array([])
      self._metadata = []
    else:
      # Directly sets dataset and metadata.
      self._dataset = dataset
      self._metadata = metadata

  def __len__(self):
    return len(self.dataset)

  @property
  def dataset(self) -> np.ndarray:
    """Gets the dataset.

    Due to performance consideration, we don't return a copy, but the returned
    `self._dataset` should never be changed.
    """
    if self._cache_dataset_list:
      # Concatenates the `self._dataset` and the datasets in
      # `self._cache_dataset_list`.
      if self._dataset.size > 0:
        dataset_list = [self._dataset] + self._cache_dataset_list
      else:
        dataset_list = self._cache_dataset_list

      self._dataset = np.vstack(dataset_list)
      self._cache_dataset_list = []
    return self._dataset

  @property
  def metadata(self) -> List[AnyStr]:
    """Gets the metadata."""
    return self._metadata

  @property
  def embedder_path(self) -> Optional[str]:
    """Gets the path to the TFLite Embedder model file."""
    return self._embedder_path

  def append(self, data_loader: "DataLoader") -> None:
    """Appends the dataset.

    Don't check if embedders from the two data loader are the same in this
    function. Users are responsible to keep the embedder identical.

    Args:
      data_loader: The data loader in which the data will be appended.
    """
    if self._dataset.shape[1] != data_loader.dataset.shape[1]:
      raise ValueError("The embedding dimension must be the same.")

    # Appends the array.
    self._cache_dataset_list.append(data_loader.dataset)
    self._metadata = self._metadata + data_loader.metadata
