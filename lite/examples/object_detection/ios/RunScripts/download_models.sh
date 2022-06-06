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
MOBILENETV1_SSD_URL="https://tfhub.dev/tensorflow/lite-model/ssd_mobilenet_v1/1/metadata/2?lite-format=tflite"
EFFICIENTDET_LITE0_URL="https://tfhub.dev/tensorflow/lite-model/efficientdet/lite0/detection/metadata/1?lite-format=tflite"
EFFICIENTDET_LITE1_URL="https://tfhub.dev/tensorflow/lite-model/efficientdet/lite1/detection/metadata/1?lite-format=tflite"
EFFICIENTDET_LITE2_URL="https://tfhub.dev/tensorflow/lite-model/efficientdet/lite2/detection/metadata/1?lite-format=tflite"

MOBILENETV1_SSD_NAME="ssd_mobilenet_v1.tflite"
EFFICIENTDET_LITE0_NAME="efficientdet_lite0.tflite"
EFFICIENTDET_LITE1_NAME="efficientdet_lite1.tflite"
EFFICIENTDET_LITE2_NAME="efficientdet_lite2.tflite"

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

if [ -f ../ObjectDetection/TFLite/${MOBILENETV1_SSD_NAME} ]
then
echo "File ${MOBILENETV1_SSD_NAME} exists."
else
has_download=true
download "${MOBILENETV1_SSD_URL}" "${DOWNLOADS_DIR}/models" "${MOBILENETV1_SSD_NAME}"
file ${DOWNLOADS_DIR}/models
fi

if [ -f ../ObjectDetection/TFLite/${EFFICIENTDET_LITE0_NAME} ]
then
echo "File ${EFFICIENTDET_LITE0_NAME} exists."
else
has_download=true
download "${EFFICIENTDET_LITE0_URL}" "${DOWNLOADS_DIR}/models" "${EFFICIENTDET_LITE0_NAME}"
file ${DOWNLOADS_DIR}/models
fi

if [ -f ../ObjectDetection/TFLite/${EFFICIENTDET_LITE1_NAME} ]
then
echo "File ${EFFICIENTDET_LITE1_NAME} exists."
else
has_download=true
download "${EFFICIENTDET_LITE1_URL}" "${DOWNLOADS_DIR}/models" "${EFFICIENTDET_LITE1_NAME}"
file ${DOWNLOADS_DIR}/models
fi

if [ -f ../ObjectDetection/TFLite/${EFFICIENTDET_LITE2_NAME} ]
then
echo "File ${EFFICIENTDET_LITE2_NAME} exists."
else
has_download=true
download "${EFFICIENTDET_LITE2_URL}" "${DOWNLOADS_DIR}/models" "${EFFICIENTDET_LITE2_NAME}"
file ${DOWNLOADS_DIR}/models
fi

if ${has_download}
then
cp ${DOWNLOADS_DIR}/models/* ../ObjectDetection/TFLite
rm -rf ${DOWNLOADS_DIR}
fi

