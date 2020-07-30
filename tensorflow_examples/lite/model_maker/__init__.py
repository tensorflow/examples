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
"""TFLite Model Maker library."""

from tensorflow_examples.lite.model_maker.core.data_util import image_dataloader
from tensorflow_examples.lite.model_maker.core.data_util import text_dataloader
from tensorflow_examples.lite.model_maker.core.data_util.image_dataloader import ImageClassifierDataLoader
from tensorflow_examples.lite.model_maker.core.data_util.text_dataloader import QuestionAnswerDataLoader
from tensorflow_examples.lite.model_maker.core.data_util.text_dataloader import TextClassifierDataLoader

from tensorflow_examples.lite.model_maker.core.task import configs
from tensorflow_examples.lite.model_maker.core.task import image_classifier
from tensorflow_examples.lite.model_maker.core.task import model_spec
from tensorflow_examples.lite.model_maker.core.task import question_answer
from tensorflow_examples.lite.model_maker.core.task import text_classifier
