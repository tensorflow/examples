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
MODELS_PREDICT_QUANTIZED_URL="https://storage.googleapis.com/download.tensorflow.org/models/tflite/arbitrary_style_transfer/style_predict_quantized_256.tflite"
MODELS_TRANSFER_QUANTIZED_URL="https://storage.googleapis.com/download.tensorflow.org/models/tflite/arbitrary_style_transfer/style_transfer_quantized_384.tflite"
MODELS_PREDICT_URL="https://storage.googleapis.com/download.tensorflow.org/models/tflite/arbitrary_style_transfer/style_predict_f16_256.tflite"
MODELS_TRANSFER_URL="https://storage.googleapis.com/download.tensorflow.org/models/tflite/arbitrary_style_transfer/style_transfer_f16_384.tflite"
DOWNLOADS_DIR=$(mktemp -d)

cd $SCRIPT_DIR

download() {
  local usage="Usage: download URL DIR"
  local url="${1:?${usage}}"
  local dir="${2:?${usage}}"
  echo "downloading ${url}" >&2
  mkdir -p "${dir}"

  cd ${dir} && { curl -L -O ${url} ; cd -; }
}

if [ ! -f ../StyleTransfer/Model/style_predict_quantized_256.tflite ]
then
download "${MODELS_PREDICT_QUANTIZED_URL}" "${DOWNLOADS_DIR}/models"
fi

if [ ! -f ../StyleTransfer/Model/style_transfer_quantized_384.tflite ]
then
download "${MODELS_TRANSFER_QUANTIZED_URL}" "${DOWNLOADS_DIR}/models"
fi

if [ ! -f ../StyleTransfer/Model/style_predict_f16_256.tflite ]
then
download "${MODELS_PREDICT_URL}" "${DOWNLOADS_DIR}/models"
fi

if [ ! -f ../StyleTransfer/Model/style_transfer_f16_384.tflite ]
then
download "${MODELS_TRANSFER_URL}" "${DOWNLOADS_DIR}/models"
fi

if [ -d ${DOWNLOADS_DIR}/models ]
then
cp ${DOWNLOADS_DIR}/models/* ../StyleTransfer/Model
fi
