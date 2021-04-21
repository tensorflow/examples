# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Public APIs for TFLite Model Maker, a transfer learning library to train custom TFLite models."""

from tflite_model_maker import audio_classifier
from tflite_model_maker import config
from tflite_model_maker import image_classifier
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector
from tflite_model_maker import question_answer
from tflite_model_maker import recommendation
from tflite_model_maker import text_classifier

__version__ = '0.2.6'
