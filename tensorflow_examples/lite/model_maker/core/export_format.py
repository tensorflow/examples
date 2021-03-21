# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Export format such as saved_model / tflite."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import enum


@enum.unique
class ExportFormat(enum.Enum):
  TFLITE = "TFLITE"
  SAVED_MODEL = "SAVED_MODEL"
  LABEL = "LABEL"
  VOCAB = "VOCAB"
  TFJS = "TFJS"


@enum.unique
class QuantizationType(enum.Enum):
  INT8 = "INT8"
  FP16 = "FP16"
  FP32 = "FP32"
  DYNAMIC = "DYNAMIC"
