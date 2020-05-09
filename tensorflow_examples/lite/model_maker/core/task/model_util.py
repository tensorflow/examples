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
"""Utilities for keras models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def set_batch_size(model, batch_size):
  """Sets batch size for the model."""
  for model_input in model.inputs:
    new_shape = [batch_size] + model_input.shape[1:]
    model_input.set_shape(new_shape)
