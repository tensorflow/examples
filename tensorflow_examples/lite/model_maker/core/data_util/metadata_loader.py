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
"""MetadataLoader to load metadata for each input data."""

import enum
import os
from typing import AnyStr, Callable

from tensorflow_examples.lite.model_maker.core.api.api_util import mm_export


@mm_export("searcher.MetadataType")
@enum.unique
class MetadataType(enum.Enum):
  FROM_FILE_NAME = 1
  FROM_DAT_FILE = 2


class MetadataLoader(object):
  """MetadataLoader class to load metadata for each input data."""

  def __init__(self, func: Callable[..., AnyStr]) -> None:
    self._func = func

  def load(self, path: str, **kwargs) -> AnyStr:
    """Loads metadata from input data path."""
    return self._func(path, **kwargs)

  @classmethod
  def from_file_name(cls) -> "MetadataLoader":
    """Loads the file name as metadata."""

    def _load_metadata(path):
      return os.path.splitext(os.path.basename(path))[0]

    return cls(_load_metadata)

  @classmethod
  def from_dat_file(cls) -> "MetadataLoader":
    """Loads the metadata from .dat file.

    Reads metadata from a file assumed to be stored next to the (image, audio or
    text) input file, with the same basename and a different extension ".dat",
    e.g: if input file is "/path/to/foo.jpg", read metadata from
    "/path/to/foo.dat". When loading metadata, user can pass `mode` parameter
    which is a string that defines which mode you want to open the file in: "r"
    (text metadata) and "rb" (binary metadata) are supported.

    Returns:
      MetadataLoader object.
    """

    def _load_metadata(path, mode="r"):
      if mode != "r" and mode != "rb":
        raise ValueError(
            "`mode` must be \"r\" or \"rb\" to read text or binary metadata.")

      metadata_file_path = os.path.splitext(path)[0] + ".dat"
      with open(metadata_file_path, mode) as f:
        return f.read()

    return cls(_load_metadata)
