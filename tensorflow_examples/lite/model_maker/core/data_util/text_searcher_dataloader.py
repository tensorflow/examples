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
"""Text DataLoader for Searcher task."""

import csv
import logging
import os

import numpy as np
from tensorflow_examples.lite.model_maker.core.api.api_util import mm_export
from tensorflow_examples.lite.model_maker.core.data_util import searcher_dataloader
from tensorflow_lite_support.python.task.core.proto import base_options_pb2
from tensorflow_lite_support.python.task.processor.proto import embedding_options_pb2
from tensorflow_lite_support.python.task.text import text_embedder

_BaseOptions = base_options_pb2.BaseOptions


@mm_export("searcher.TextDataLoader")
class DataLoader(searcher_dataloader.DataLoader):
  """DataLoader class for Text Searcher."""

  def __init__(self, embedder: text_embedder.TextEmbedder) -> None:
    """Initializes DataLoader for Image Searcher task.

    Args:
      embedder: Embedder to generate embedding from raw input image.
    """
    self._embedder = embedder
    super().__init__(embedder_path=embedder.options.base_options.file_name)

  @classmethod
  def create(cls,
             text_embedder_path: str,
             l2_normalize: bool = False) -> "DataLoader":
    """Creates DataLoader for the Text Searcher task.

    Args:
      text_embedder_path: Path to the ".tflite" text embedder model. case and L2
        norm is thus achieved through TF Lite inference.
      l2_normalize: Whether to normalize the returned feature vector with L2
        norm. Use this option only if the model does not already contain a
        native L2_NORMALIZATION TF Lite Op. In most cases, this is already the
        case and L2 norm is thus achieved through TF Lite inference.

    Returns:
      DataLoader object created for the Text Searcher task.
    """
    # Creates TextEmbedder.
    text_embedder_path = os.path.abspath(text_embedder_path)
    base_options = _BaseOptions(file_name=text_embedder_path)
    embedding_options = embedding_options_pb2.EmbeddingOptions(
        l2_normalize=l2_normalize)
    options = text_embedder.TextEmbedderOptions(
        base_options=base_options, embedding_options=embedding_options)
    embedder = text_embedder.TextEmbedder.create_from_options(options)

    return cls(embedder)

  def load_from_csv(self,
                    path: str,
                    text_column: str,
                    metadata_column: str,
                    delimiter: str = ",",
                    quotechar: str = "\"") -> None:
    """Loads text data from csv file that includes a "header" line with titles.

    Users can load text from different csv files one by one. For instance,
    ```
    # Creates data_loader instance.
    data_loader = text_searcher_dataloader.DataLoader.create(tflite_path)

    # Loads text, first from `text_path1` and secondly from `text_path2`.
    data_loader.load_from_csv(
        text_path1, text_column='text', metadata_column='metadata')
    data_loader.load_from_csv(
        text_path2, text_column='text', metadata_column='metadata')
    ```

    Args:
      path: Text csv file path to be loaded.
      text_column: Column name for input text.
      metadata_column: Column name for user metadata associated with each input
        text.
      delimiter: Character used to separate fields.
      quotechar: Character used to quote fields containing special characters.
    """
    embedding_list = []
    metadata_list = []

    i = 0
    # Reads the text and metadata one by one from the csv file.
    with open(path, "r") as f:
      reader = csv.DictReader(f, delimiter=delimiter, quotechar=quotechar)
      for line in reader:
        text, metadata = line[text_column], line[metadata_column]
        try:
          embedding = self._embedder.embed(
              text).embeddings[0].feature_vector.value
        except (ValueError, RuntimeError) as e:
          logging.warning("Can't get the embedding of %s with the error %s",
                          text, e)
          continue

        embedding_list.append(embedding)
        metadata_list.append(metadata)

        i += 1
        if i % 1000 == 0:
          logging.info("Processed %d text strings.", i)

    cache_dataset = np.stack(embedding_list)
    self._cache_dataset_list.append(cache_dataset)
    self._metadata = self._metadata + metadata_list
