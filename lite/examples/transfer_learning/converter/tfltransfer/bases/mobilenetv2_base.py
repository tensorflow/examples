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
"""Base model configuration for MobileNetV2."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

# pylint: disable=g-bad-import-order
from tfltransfer.bases import quantizable_base
# pylint: enable=g-bad-import-order


class MobileNetV2Base(quantizable_base.QuantizableBase):
  """Base model configuration that downloads a pretrained MobileNetV2.

  The model is downloaded with weights pre-trained for ImageNet.
  The last few layers following the final DepthwiseConv are stripped
  off, and the feature vector with shape (7, 7, 1280) is used as
  the output.
  """

  def __init__(self,
               image_size=224,
               alpha=1.0,
               quantize=False,
               representative_dataset=None):
    """Constructs a MobileNetV2 base model configuration.

    Args:
      image_size: width and height of the input to the model.
      alpha: controls the width of the network. This is known as the width
        multiplier in the MobileNetV2 paper.
      quantize: whether the model weights should be quantized.
      representative_dataset: generator that yields representative data for full
        integer quantization. If None, hybrid quantization is performed.
    """
    super(MobileNetV2Base, self).__init__(quantize, representative_dataset)
    self._image_size = image_size
    self._alpha = alpha

  def prepare_converter(self):
    """Prepares an initial configuration of a TFLiteConverter."""
    model = tf.keras.applications.MobileNetV2(
        input_shape=(self._image_size, self._image_size, 3),
        alpha=self._alpha,
        include_top=False,
        weights='imagenet')
    return tf.lite.TFLiteConverter.from_keras_model(model)

  def bottleneck_shape(self):
    """Reads the shape of the bottleneck produced by the model."""
    return (7, 7, 1280)
