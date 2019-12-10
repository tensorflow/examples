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
"""Base model abstract base class that handles quantization."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

import tensorflow as tf


class QuantizableBase(object):
  """Base model abstract base class that handles quantization."""

  __metaclass__ = abc.ABCMeta

  def __init__(self, quantize, representative_dataset):
    """Constructs a QuantizableBase instance.

    Args:
      quantize: whether the model weights should be quantized.
      representative_dataset: generator that yields representative data for full
        integer quantization. If None, hybrid quantization is performed.

    Raises:
      ValueError: when an unsupported combination of arguments is used.
    """
    if representative_dataset and not quantize:
      raise ValueError(
          'representative_dataset cannot be specified when quantize is False.')
    self._quantize = quantize
    self._representative_dataset = representative_dataset

  @abc.abstractmethod
  def prepare_converter(self):
    """Prepares an initial configuration of a TFLiteConverter.

    Quantization parameters are possibly added to this configuration
    when the model is generated.

    Returns:
      TFLiteConverter instance.
    """

  def tflite_model(self):
    converter = self.prepare_converter()
    if self._quantize and self._representative_dataset:
      converter.optimizations = [tf.lite.Optimize.DEFAULT]
      converter.representative_dataset = self._representative_dataset
    elif self._quantize:
      converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]

    return converter.convert()
