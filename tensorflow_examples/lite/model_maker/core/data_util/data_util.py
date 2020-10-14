# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Custom classification model that is already retained by data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf
from tensorflow_examples.lite.model_maker.core import compat


def generate_elements(ds):
  """Generates elements from `tf.data.dataset`."""
  if compat.get_tf_behavior() == 2:
    for element in ds.as_numpy_iterator():
      yield element
  else:
    iterator = tf.compat.v1.data.make_one_shot_iterator(ds)
    next_element = iterator.get_next()
    with tf.compat.v1.Session() as sess:
      try:
        while True:
          yield sess.run(next_element)
      except tf.errors.OutOfRangeError:
        return
