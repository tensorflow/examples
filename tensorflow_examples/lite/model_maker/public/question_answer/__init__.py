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
"""APIs to train a model that can answer questions based on a predefined text.

Task guide:
https://www.tensorflow.org/lite/tutorials/model_maker_question_answer.
"""

from tensorflow_examples.lite.model_maker.core.data_util.text_dataloader import QuestionAnswerDataLoader as DataLoader
from tensorflow_examples.lite.model_maker.core.task.model_spec.text_spec import BertQAModelSpec as BertQaSpec
from tensorflow_examples.lite.model_maker.core.task.model_spec.text_spec import mobilebert_qa_spec as MobileBertQaSpec
from tensorflow_examples.lite.model_maker.core.task.model_spec.text_spec import mobilebert_qa_squad_spec as MobileBertQaSquadSpec
from tensorflow_examples.lite.model_maker.core.task.question_answer import create
from tensorflow_examples.lite.model_maker.core.task.question_answer import QuestionAnswer
