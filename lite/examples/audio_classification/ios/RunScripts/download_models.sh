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
YAMNET_URL="https://tfhub.dev/google/lite-model/yamnet/classification/tflite/1?lite-format=tflite"
SPEECH_COMMANDS_URL="https://storage.googleapis.com/download.tensorflow.org/models/tflite/speech_commands.tflite"

YAMNET_NAME="yamnet.tflite"
SPEECH_COMMANDS_NAME="speech_commands.tflite"

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

if [ -f ../AudioClassification/TFLite/${YAMNET_NAME} ]
then
echo "File ${YAMNET_NAME} exists."
else
has_download=true
download "${YAMNET_URL}" "${DOWNLOADS_DIR}/models" "${YAMNET_NAME}"
file ${DOWNLOADS_DIR}/models
fi

if [ -f ../AudioClassification/TFLite/${SPEECH_COMMANDS_NAME} ]
then
echo "File ${SPEECH_COMMANDS_NAME} exists."
else
has_download=true
download "${SPEECH_COMMANDS_URL}" "${DOWNLOADS_DIR}/models" "${SPEECH_COMMANDS_NAME}"
file ${DOWNLOADS_DIR}/models
fi

if ${has_download}
then
cp ${DOWNLOADS_DIR}/models/* ../AudioClassification/TFLite
rm -rf ${DOWNLOADS_DIR}
fi

