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
"""Audio model specification."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tempfile

import tensorflow as tf


class BaseSpec(abc.ABC):
  """Base model spec for audio classification."""

  compat_tf_versions = (2,)

  def __init__(self, model_dir=None, strategy=None):
    self.model_dir = model_dir
    if not model_dir:
      self.model_dir = tempfile.mkdtemp()
    tf.compat.v1.logging.info('Checkpoints are stored in %s', self.model_dir)
    self.strategy = strategy or tf.distribute.get_strategy()

    self.expected_waveform_len = 44032
    self.target_sample_rate = 44100
    self.snippet_duration_sec = 1.

  @abc.abstractmethod
  def create_model(self):
    pass

  @abc.abstractmethod
  def run_classifier(self, model, train_input_fn, validation_input_fn, epochs,
                     steps_per_epoch, validation_steps):
    pass

  # Default dummy augmentation that will be applied to train samples.
  def data_augmentation(self, x):
    return x

  # Default dummy preprocessing that will be applied to all data samples.
  def preprocess(self, x):
    return x
