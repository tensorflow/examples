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
"""Recommendation model specification."""

import tensorflow as tf  # pylint: disable=unused-import
from tensorflow_examples.lite.model_maker.core import compat
from tensorflow_examples.lite.model_maker.core.api import mm_export
from tensorflow_examples.lite.model_maker.core.data_util import recommendation_config
from tensorflow_examples.lite.model_maker.third_party.recommendation.ml.model import recommendation_model as _model


@mm_export('recommendation.ModelSpec')
class RecommendationSpec(object):
  """Recommendation model spec."""

  compat_tf_versions = compat.get_compat_tf_versions(2)

  def __init__(self, input_spec: recommendation_config.InputSpec,
               model_hparams: recommendation_config.ModelHParams):
    """Initialize spec.

    Args:
      input_spec: InputSpec, specify data format for input and embedding.
      model_hparams: ModelHParams, specify hparams for model achitecture.
    """
    self.input_spec = input_spec
    self.model_hparams = model_hparams

  def create_model(self):
    """Creates recommendation model based on params.

    Returns:
      Keras model.
    """
    return _model.RecommendationModel(self.input_spec, self.model_hparams)

  def get_default_quantization_config(self):
    """Gets the default quantization configuration."""
    return None
