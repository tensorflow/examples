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
"""APIs to train an object detection model."""

from tensorflow_examples.lite.model_maker.core.data_util.object_detector_dataloader import DataLoader
from tensorflow_examples.lite.model_maker.core.task.model_spec.object_detector_spec import efficientdet_lite0_spec as EfficientDetLite0Spec
from tensorflow_examples.lite.model_maker.core.task.model_spec.object_detector_spec import efficientdet_lite1_spec as EfficientDetLite1Spec
from tensorflow_examples.lite.model_maker.core.task.model_spec.object_detector_spec import efficientdet_lite2_spec as EfficientDetLite2Spec
from tensorflow_examples.lite.model_maker.core.task.model_spec.object_detector_spec import efficientdet_lite3_spec as EfficientDetLite3Spec
from tensorflow_examples.lite.model_maker.core.task.model_spec.object_detector_spec import efficientdet_lite4_spec as EfficientDetLite4Spec
from tensorflow_examples.lite.model_maker.core.task.model_spec.object_detector_spec import EfficientDetModelSpec as EfficientDetSpec
from tensorflow_examples.lite.model_maker.core.task.object_detector import create
from tensorflow_examples.lite.model_maker.core.task.object_detector import ObjectDetector
