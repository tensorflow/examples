#!/bin/bash
# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================

# Download TF Lite model from the internet if it does not exist.

MODEL_FOLDER="GestureClassification/Model"
TFLITE_MODEL="${MODEL_FOLDER}/model.tflite"
TFLITE_LABELS="${MODEL_FOLDER}/labels.txt"
TFLITE_MODEL_REMOTE_PATH="https://storage.googleapis.com/download.tensorflow.org/models/tflite/task_library/gesture_classification/ios/model_metadata.tflite"
TFLITE_LABELS_REMOTE_PATH="https://storage.googleapis.com/download.tensorflow.org/models/tflite/task_library/gesture_classification/android/labels.txt"

if [ ! -f "${TFLITE_MODEL}" ]; then
    # Download model
    curl --create-dirs -o "${TFLITE_MODEL}" "${TFLITE_MODEL_REMOTE_PATH}"
    echo "INFO: Downloaded TensorFlow Lite model to $RESOURCES_DIR ."
fi

if [ ! -f "${TFLITE_LABELS}" ]; then
    # Download labels
    curl --create-dirs -o "${TFLITE_LABELS}" "${TFLITE_LABELS_REMOTE_PATH}"
    echo "INFO: Download TensorFlow Lite labels to $RESOURCES_DIR ."
fi
