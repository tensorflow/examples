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
"""Custom model that is already retained by data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc


class ClassificationModel(abc.ABC):
  """"The abstract base class that represents a Tensorflow classification model."""

  def __init__(self, data, model_export_format, model_name, shuffle,
               train_whole_model, validation_ratio, test_ratio):
    """Initialize a instance with data, deploy mode and other related parameters.

    Args:
      data: Raw data that could be splitted for training / validation / testing.
      model_export_format: Model export format such as saved_model / tflite.
      model_name: Model name.
      shuffle: Whether the data should be shuffled.
      train_whole_model: If true, the Hub module is trained together with the
        classification layer on top. Otherwise, only train the top
        classification layer.
      validation_ratio: The ratio of valid data to be splitted.
      test_ratio: The ratio of test data to be splitted.
    """
    self.data = data
    self.model_export_format = model_export_format
    self.model_name = model_name
    self.shuffle = shuffle
    self.train_whole_model = train_whole_model
    self.validation_ratio = validation_ratio
    self.test_ratio = test_ratio

  def summary(self):
    self.model.summary()

  @abc.abstractmethod
  def evaluate(self, data, batch_size=32):
    return

  @abc.abstractmethod
  def predict_topk(self, data, k=1, batch_size=32):
    return

  @abc.abstractmethod
  def export(self, **kwargs):
    return
