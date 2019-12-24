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
"""Model specification."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class ImageModelSpec(object):
  """A specification of image model."""

  input_image_shape = [224, 224]
  mean_rgb = [0, 0, 0]
  stddev_rgb = [255, 255, 255]

  def __init__(self, name, uri):
    self.name = name
    self.uri = uri

efficientnet_b0_spec = ImageModelSpec(
    name='efficientnet_b0',
    uri='https://tfhub.dev/google/efficientnet/b0/feature-vector/1')

mobilenet_v2_spec = ImageModelSpec(
    name='mobilenet_v2',
    uri='https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4')
