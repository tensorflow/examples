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
"""APIs to train an image classification model."""

from tensorflow_examples.lite.model_maker.core.data_util.image_dataloader import ImageClassifierDataLoader as DataLoader
from tensorflow_examples.lite.model_maker.core.task.image_classifier import create
from tensorflow_examples.lite.model_maker.core.task.image_classifier import ImageClassifier
from tensorflow_examples.lite.model_maker.core.task.model_spec.image_spec import efficientnet_lite0_spec as EfficientNetLite0Spec
from tensorflow_examples.lite.model_maker.core.task.model_spec.image_spec import efficientnet_lite1_spec as EfficientNetLite1Spec
from tensorflow_examples.lite.model_maker.core.task.model_spec.image_spec import efficientnet_lite2_spec as EfficientNetLite2Spec
from tensorflow_examples.lite.model_maker.core.task.model_spec.image_spec import efficientnet_lite3_spec as EfficientNetLite3Spec
from tensorflow_examples.lite.model_maker.core.task.model_spec.image_spec import efficientnet_lite4_spec as EfficientNetLite4Spec
from tensorflow_examples.lite.model_maker.core.task.model_spec.image_spec import ImageModelSpec as ModelSpec
from tensorflow_examples.lite.model_maker.core.task.model_spec.image_spec import mobilenet_v2_spec as MobileNetV2Spec
from tensorflow_examples.lite.model_maker.core.task.model_spec.image_spec import resnet_50_spec as Resnet50Spec
