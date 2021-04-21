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
"""APIs to train an on-device recommendation model."""

from tensorflow_examples.lite.model_maker.core.data_util.recommendation_dataloader import RecommendationDataLoader as DataLoader
from tensorflow_examples.lite.model_maker.core.task.model_spec.recommendation_spec import recommendation_bow_spec as BowSpec
from tensorflow_examples.lite.model_maker.core.task.model_spec.recommendation_spec import recommendation_cnn_spec as CnnSpec
from tensorflow_examples.lite.model_maker.core.task.model_spec.recommendation_spec import recommendation_rnn_spec as RnnSpec
from tensorflow_examples.lite.model_maker.core.task.model_spec.recommendation_spec import RecommendationSpec as ModelSpec
from tensorflow_examples.lite.model_maker.core.task.recommendation import create
from tensorflow_examples.lite.model_maker.core.task.recommendation import Recommendation
