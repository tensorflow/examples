#!/bin/bash

if [ $# -eq 0 ]; then
  DATA_DIR="./"
else
  DATA_DIR="$1"
fi

# Install Python dependencies
python3 -m pip install -r requirements.txt

# Download TF Lite models with metadata.
FILE=${DATA_DIR}/yamnet.tflite
if [ ! -f "$FILE" ]; then
  curl \
    -L 'https://storage.googleapis.com/download.tensorflow.org/models/tflite/task_library/audio_classification/rpi/lite-model_yamnet_classification_tflite_1.tflite' \
    -o ${FILE}
fi

FILE=${DATA_DIR}/yamnet_edgetpu.tflite
if [ ! -f "$FILE" ]; then
  curl \
    -L 'https://storage.googleapis.com/download.tensorflow.org/models/tflite/task_library/audio_classification/rpi/coral-model_yamnet_classification_coral_1.tflite' \
    -o ${FILE}
fi
echo -e "Downloaded files are in ${DATA_DIR}"
