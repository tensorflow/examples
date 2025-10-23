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


MODEL_DIR="PoseEstimation/ML/Models"

# Download TF Lite models
FILE=${MODEL_DIR}/posenet.tflite
if [ ! -f "$FILE" ]; then
  curl \
    -L 'https://storage.googleapis.com/download.tensorflow.org/models/tflite/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite' \
    -o ${FILE}
fi

FILE=${MODEL_DIR}/movenet_lightning.tflite
GZ_FILE=${MODEL_DIR}/movenet_lightning.tar.gz
if [ ! -f "$FILE" ]; then
  curl \
    -L 'https://www.kaggle.com/api/v1/models/google/movenet/tfLite/multipose-lightning-tflite-float16/1/download' \
    -o ${GZ_FILE}
  tar -xzf ${GZ_FILE} -C ${MODEL_DIR}
  rm ${GZ_FILE}
  mv ${MODEL_DIR}/1.tflite ${FILE}
fi

FILE=${MODEL_DIR}/movenet_thunder.tflite
GZ_FILE=${MODEL_DIR}/movenet_thunder.tar.gz
if [ ! -f "$FILE" ]; then
  curl \
    -L 'https://www.kaggle.com/api/v1/models/google/movenet/tfLite/singlepose-thunder-tflite-float16/1/download' \
    -o ${GZ_FILE}
  tar -xzf ${GZ_FILE} -C ${MODEL_DIR}
  rm ${GZ_FILE}
  mv ${MODEL_DIR}/4.tflite ${FILE}
fi

echo -e "Downloaded files are in ${MODEL_DIR}"



