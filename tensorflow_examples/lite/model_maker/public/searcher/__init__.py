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
"""APIs to create the searcher model.

Task guide:
https://www.tensorflow.org/lite/tutorials/model_maker_text_searcher.
"""

from tensorflow_examples.lite.model_maker.core.data_util.image_searcher_dataloader import DataLoader as ImageDataLoader
from tensorflow_examples.lite.model_maker.core.data_util.metadata_loader import MetadataType
from tensorflow_examples.lite.model_maker.core.data_util.searcher_dataloader import DataLoader
from tensorflow_examples.lite.model_maker.core.data_util.text_searcher_dataloader import DataLoader as TextDataLoader
from tensorflow_examples.lite.model_maker.core.task.searcher import ExportFormat
from tensorflow_examples.lite.model_maker.core.task.searcher import ScaNNOptions
from tensorflow_examples.lite.model_maker.core.task.searcher import ScoreAH
from tensorflow_examples.lite.model_maker.core.task.searcher import ScoreBruteForce
from tensorflow_examples.lite.model_maker.core.task.searcher import Searcher
from tensorflow_examples.lite.model_maker.core.task.searcher import Tree
