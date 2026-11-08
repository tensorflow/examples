#!/bin/bash

if [ $# -eq 0 ]; then
  DATA_DIR="./"
else
  DATA_DIR="$1"
fi

# Install Python dependencies.
python3 -m pip install pip --upgrade
python3 -m pip install -r requirements.txt

# Download TF Lite model with metadata.
FILE=${DATA_DIR}/efficientnet_lite0.tflite
if [ ! -f "$FILE" ]; then
  curl \
    -L 'https://storage.googleapis.com/download.tensorflow.org/models/tflite/task_library/image_classification/rpi/lite-model_efficientnet_lite0_uint8_2.tflite' \
    -o ${FILE}
fi

FILE=${DATA_DIR}/efficientnet_lite0_edgetpu.tflite
if [ ! -f "$FILE" ]; then
  curl \
    -L 'https://storage.googleapis.com/download.tensorflow.org/models/tflite/task_library/image_classification/rpi/efficientnet_lite0_edgetpu.tflite' \
    -o ${FILE}
fi
echo -e "Downloaded files are in ${DATA_DIR}"
