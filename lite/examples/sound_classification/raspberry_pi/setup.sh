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
    -L 'https://tfhub.dev/google/lite-model/yamnet/classification/tflite/1?lite-format=tflite' \
    -o ${FILE}
fi

FILE=${DATA_DIR}/yamnet_edgetpu.tflite
if [ ! -f "$FILE" ]; then
  curl \
    -L 'https://tfhub.dev/google/coral-model/yamnet/classification/coral/1?coral-format=tflite' \
    -o ${FILE}
fi
echo -e "Downloaded files are in ${DATA_DIR}"