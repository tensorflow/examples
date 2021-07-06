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
# pylint: disable=g-bad-import-order,redefined-builtin
"""APIs to train an audio classification model.

Tutorial:
https://colab.research.google.com/github/googlecodelabs/odml-pathways/blob/main/audio_classification/colab/model_maker_audio_colab.ipynb

Demo code:
https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/demo/audio_classification_demo.py
"""

from tensorflow_examples.lite.model_maker.core.data_util.audio_dataloader import DataLoader
from tensorflow_examples.lite.model_maker.core.task.audio_classifier import AudioClassifier
from tensorflow_examples.lite.model_maker.core.task.audio_classifier import create
from tensorflow_examples.lite.model_maker.core.task.model_spec.audio_spec import BrowserFFTSpec as BrowserFftSpec
from tensorflow_examples.lite.model_maker.core.task.model_spec.audio_spec import YAMNetSpec as YamNetSpec
