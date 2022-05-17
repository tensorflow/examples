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
"""Packages included in API."""

# pylint: disable=unused-import
from tensorflow_examples.lite.model_maker.core import compat
from tensorflow_examples.lite.model_maker.core.data_util import audio_dataloader
from tensorflow_examples.lite.model_maker.core.data_util import image_dataloader
from tensorflow_examples.lite.model_maker.core.data_util import object_detector_dataloader
from tensorflow_examples.lite.model_maker.core.data_util import text_dataloader

from tensorflow_examples.lite.model_maker.core.data_util.audio_dataloader import DataLoader as AudioDataLoader
from tensorflow_examples.lite.model_maker.core.data_util.image_dataloader import ImageClassifierDataLoader
from tensorflow_examples.lite.model_maker.core.data_util.object_detector_dataloader import DataLoader as ObjectDetectorDataloader
from tensorflow_examples.lite.model_maker.core.data_util.recommendation_dataloader import RecommendationDataLoader
from tensorflow_examples.lite.model_maker.core.data_util.text_dataloader import QuestionAnswerDataLoader
from tensorflow_examples.lite.model_maker.core.data_util.text_dataloader import TextClassifierDataLoader

from tensorflow_examples.lite.model_maker.core.export_format import ExportFormat

from tensorflow_examples.lite.model_maker.core.task import audio_classifier
from tensorflow_examples.lite.model_maker.core.task import configs
from tensorflow_examples.lite.model_maker.core.task import image_classifier
from tensorflow_examples.lite.model_maker.core.task import model_spec
from tensorflow_examples.lite.model_maker.core.task import object_detector
from tensorflow_examples.lite.model_maker.core.task import question_answer
from tensorflow_examples.lite.model_maker.core.task import recommendation
from tensorflow_examples.lite.model_maker.core.task import text_classifier

from tensorflow_examples.lite.model_maker.core.data_util import image_searcher_dataloader
from tensorflow_examples.lite.model_maker.core.data_util import metadata_loader
from tensorflow_examples.lite.model_maker.core.data_util import searcher_dataloader
from tensorflow_examples.lite.model_maker.core.data_util import text_searcher_dataloader
from tensorflow_examples.lite.model_maker.core.task import searcher
