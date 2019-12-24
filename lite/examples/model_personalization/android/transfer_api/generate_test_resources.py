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
"""Generate helper TFLite models that are used for tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

SOFTMAX_INITIALIZE_ONES_PATH = './src/androidTest/assets/model/softmax_initialize_ones.tflite'


@tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.float32)])
def initial_params(zero):
  ws = tf.fill((7 * 7 * 1280, 5), zero + 1)
  bs = tf.fill((5,), zero + 1)
  return ws, bs


converter = tf.lite.TFLiteConverter.from_concrete_functions(
    [initial_params.get_concrete_function()])
model_lite = converter.convert()
with open(SOFTMAX_INITIALIZE_ONES_PATH, 'wb') as f:
  f.write(model_lite)
