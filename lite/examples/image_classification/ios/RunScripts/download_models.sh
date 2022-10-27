#!/bin/bash
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
# ==============================================================================

set -ex

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EFFICIENTNET_LITE0_URL="https://storage.googleapis.com/download.tensorflow.org/models/tflite/task_library/image_classification/ios/lite-model_efficientnet_lite0_uint8_2.tflite"
EFFICIENTNET_LITE1_URL="https://storage.googleapis.com/download.tensorflow.org/models/tflite/task_library/image_classification/ios/lite-model_efficientnet_lite1_uint8_2.tflite"
EFFICIENTNET_LITE2_URL="https://storage.googleapis.com/download.tensorflow.org/models/tflite/task_library/image_classification/ios/lite-model_efficientnet_lite2_uint8_2.tflite"
EFFICIENTNET_LITE3_URL="https://storage.googleapis.com/download.tensorflow.org/models/tflite/task_library/image_classification/ios/lite-model_efficientnet_lite3_uint8_2.tflite"
EFFICIENTNET_LITE4_URL="https://storage.googleapis.com/download.tensorflow.org/models/tflite/task_library/image_classification/ios/lite-model_efficientnet_lite4_uint8_2.tflite"

EFFICIENTNET_LITE0_NAME="efficientnet_lite0.tflite"
EFFICIENTNET_LITE1_NAME="efficientnet_lite1.tflite"
EFFICIENTNET_LITE2_NAME="efficientnet_lite2.tflite"
EFFICIENTNET_LITE3_NAME="efficientnet_lite3.tflite"
EFFICIENTNET_LITE4_NAME="efficientnet_lite4.tflite"

DOWNLOADS_DIR=$(mktemp -d)

cd "$SCRIPT_DIR"

download() {
  local usage="Usage: download_and_extract URL DIR"
  local url="${1:?${usage}}"
  local dir="${2:?${usage}}"
  local name="${3:?${usage}}"
  echo "downloading ${url}" >&2
  mkdir -p "${dir}"
  tempdir=$(mktemp -d)

  curl -L ${url} > ${tempdir}/${name}
  cp -R ${tempdir}/* ${dir}/
  rm -rf ${tempdir}
}

has_download=false

if [ -f ../ImageClassification/TFLite/${EFFICIENTNET_LITE0_NAME} ]
then
echo "File ${EFFICIENTNET_LITE0_NAME} exists."
else
has_download=true
download "${EFFICIENTNET_LITE0_URL}" "${DOWNLOADS_DIR}/models" "${EFFICIENTNET_LITE0_NAME}"
file ${DOWNLOADS_DIR}/models
fi

if [ -f ../ImageClassification/TFLite/${EFFICIENTNET_LITE1_NAME} ]
then
echo "File ${EFFICIENTNET_LITE1_NAME} exists."
else
has_download=true
download "${EFFICIENTNET_LITE1_URL}" "${DOWNLOADS_DIR}/models" "${EFFICIENTNET_LITE1_NAME}"
file ${DOWNLOADS_DIR}/models
fi

if [ -f ../ImageClassification/TFLite/${EFFICIENTNET_LITE2_NAME} ]
then
echo "File ${EFFICIENTNET_LITE2_NAME} exists."
else
has_download=true
download "${EFFICIENTNET_LITE2_URL}" "${DOWNLOADS_DIR}/models" "${EFFICIENTNET_LITE2_NAME}"
file ${DOWNLOADS_DIR}/models
fi

if [ -f ../ImageClassification/TFLite/${EFFICIENTNET_LITE3_NAME} ]
then
echo "File ${EFFICIENTNET_LITE3_NAME} exists."
else
has_download=true
download "${EFFICIENTNET_LITE3_URL}" "${DOWNLOADS_DIR}/models" "${EFFICIENTNET_LITE3_NAME}"
file ${DOWNLOADS_DIR}/models
fi

if [ -f ../ImageClassification/TFLite/${EFFICIENTNET_LITE4_NAME} ]
then
echo "File ${EFFICIENTNET_LITE4_NAME} exists."
else
has_download=true
download "${EFFICIENTNET_LITE4_URL}" "${DOWNLOADS_DIR}/models" "${EFFICIENTNET_LITE4_NAME}"
file ${DOWNLOADS_DIR}/models
fi

if ${has_download}
then
cp ${DOWNLOADS_DIR}/models/* ../ImageClassification/TFLite
rm -rf ${DOWNLOADS_DIR}
fi

