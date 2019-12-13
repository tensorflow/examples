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
"""Base model configuration that reads a specified SavedModel."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

# pylint: disable=g-bad-import-order
from tfltransfer.bases import quantizable_base
# pylint: enable=g-bad-import-order


class SavedModelBase(quantizable_base.QuantizableBase):
  """Base model configuration that reads a specified SavedModel.

  The SavedModel should contain a signature that converts
  samples to bottlenecks. This is assumed by default to be
  the main serving signature, but this can be configured.
  """

  def __init__(self,
               model_dir,
               tag=tf.saved_model.SERVING,
               signature_key='serving_default',
               quantize=False,
               representative_dataset=None):
    """Constructs a base model from a SavedModel.

    Args:
      model_dir: path to the SavedModel to load.
      tag: MetaGraphDef tag to be used.
      signature_key: signature name for the forward pass.
      quantize: whether the model weights should be quantized.
      representative_dataset: generator that yields representative data for full
        integer quantization. If None, hybrid quantization is performed.
    """
    super(SavedModelBase, self).__init__(quantize, representative_dataset)
    self._model_dir = model_dir
    self._tag = tag
    self._signature_key = signature_key

    loaded_model = tf.saved_model.load(model_dir, tags=[tag])
    signature = loaded_model.signatures[signature_key]
    self._bottleneck_shape = (
        tuple(next(signature.output_shapes.values().__iter__())[1:]))

  def prepare_converter(self):
    """Prepares an initial configuration of a TFLiteConverter."""
    return tf.lite.TFLiteConverter.from_saved_model(
        self._model_dir, signature_keys=[self._signature_key], tags=[self._tag])

  def bottleneck_shape(self):
    """Reads the shape of the bottleneck produced by the model."""
    return self._bottleneck_shape
