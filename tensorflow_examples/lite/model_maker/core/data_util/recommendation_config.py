# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Recommendation dataloader class."""

from tensorflow_examples.lite.model_maker.core.api import mm_export
from tensorflow_examples.lite.model_maker.third_party.recommendation.ml.configs import input_config_pb2
from tensorflow_examples.lite.model_maker.third_party.recommendation.ml.configs import model_config

# Shortcut for classes.
ModelHParams = model_config.ModelConfig
mm_export('recommendation.spec.ModelHParams').export_constant(
    __name__, 'ModelHParams')

InputSpec = input_config_pb2.InputConfig
Feature = input_config_pb2.Feature
FeatureGroup = input_config_pb2.FeatureGroup
FeatureType = input_config_pb2.FeatureType
EncoderType = input_config_pb2.EncoderType

mm_export('recommendation.spec.InputSpec').export_constant(
    __name__, 'InputSpec')
mm_export('recommendation.spec.Feature').export_constant(__name__, 'Feature')
mm_export('recommendation.spec.FeatureGroup').export_constant(
    __name__, 'FeatureGroup')

EncoderType.__doc__ = 'EncoderType Enum (valid: BOW, CNN, LSTM).'
mm_export('recommendation.spec.EncoderType').export_constant(
    __name__, 'EncoderType')

FeatureType.__doc__ = 'FeatureType Enum (valid: STRING, INT, FLOAT).'
mm_export('recommendation.spec.FeatureType').export_constant(
    __name__, 'FeatureType')
