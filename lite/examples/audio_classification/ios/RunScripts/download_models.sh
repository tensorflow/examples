#!/bin/bash
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
# ==============================================================================

set -ex

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_URL="https://storage.googleapis.com/download.tensorflow.org/models/tflite/sound_classification/snap_clap.tflite"
DOWNLOADS_DIR=$(mktemp -d)

cd "$SCRIPT_DIR"

download() {
  local usage="Usage: download URL DIR"
  local url="${1:?${usage}}"
  local dir="${2:?${usage}}"
  echo "downloading ${url}" >&2
  mkdir -p "${dir}"
  tempdir=$(mktemp -d)

  curl -L ${url} > ${tempdir}/sound_classification.tflite
  cp -R ${tempdir}/* ${dir}/
  rm -rf ${tempdir}
}

if [ -f ../SoundClassification/Model/sound_classification.tflite ]
then
echo "File already exists."
exit 0
fi

download "${MODELS_URL}" "${DOWNLOADS_DIR}/models"

file ${DOWNLOADS_DIR}/models

cp ${DOWNLOADS_DIR}/models/* ../SoundClassification/Model
