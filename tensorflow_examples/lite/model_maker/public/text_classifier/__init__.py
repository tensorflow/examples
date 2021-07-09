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
"""APIs to train a text classification model.

Task guide:
https://www.tensorflow.org/lite/tutorials/model_maker_text_classification.
"""

from tensorflow_examples.lite.model_maker.core.data_util.text_dataloader import TextClassifierDataLoader as DataLoader
from tensorflow_examples.lite.model_maker.core.task.model_spec.text_spec import AverageWordVecModelSpec as AverageWordVecSpec
from tensorflow_examples.lite.model_maker.core.task.model_spec.text_spec import BertClassifierModelSpec as BertClassifierSpec
from tensorflow_examples.lite.model_maker.core.task.model_spec.text_spec import mobilebert_classifier_spec as MobileBertClassifierSpec
from tensorflow_examples.lite.model_maker.core.task.text_classifier import create
from tensorflow_examples.lite.model_maker.core.task.text_classifier import TextClassifier
