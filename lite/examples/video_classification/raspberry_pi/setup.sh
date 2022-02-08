#!/bin/bash

if [ $# -eq 0 ]; then
  DATA_DIR="./"
else
  DATA_DIR="$1"
fi

# Install Python dependencies.
python3 -m pip install pip --upgrade
python3 -m pip install -r requirements.txt

# Download TF Lite models
FILE=${DATA_DIR}/movinet_a0_int8.tflite
if [ ! -f "$FILE" ]; then
  curl \
    -L 'https://tfhub.dev/tensorflow/lite-model/movinet/a0/stream/kinetics-600/classification/tflite/int8/1?lite-format=tflite' \
    -o ${FILE}
fi

echo -e "Downloaded files are in ${DATA_DIR}"