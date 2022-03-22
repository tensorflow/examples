#   Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""Class to hold model architecture configuration."""
from typing import List
import attr


@attr.s(auto_attribs=True)
class ModelConfig(object):
  """Class to hold parameters for model architecture configuration.

  Attributes:
      hidden_layer_dims: List of hidden layer dimensions.
      eval_top_k: Top k to evaluate.
      conv_num_filter_ratios: Number of filter ratios for the Conv1D layer.
      conv_kernel_size: Size of the Conv1D layer kernel size.
      lstm_num_units: Number of units for the LSTM layer.
      num_predictions: Number of predictions to return with serving mode, which
      has default value 10.
  """
  hidden_layer_dims: List[int]
  eval_top_k: List[int]
  conv_num_filter_ratios: List[int]
  conv_kernel_size: int
  lstm_num_units: int
  num_predictions: int = 10
