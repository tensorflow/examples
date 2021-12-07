#!/bin/bash

if [ $# -eq 0 ]; then
  DATA_DIR="./"
else
  DATA_DIR="$1"
fi

# Install Python dependencies.
python3 -m pip install -r requirements_pypi.txt
python3 -m pip install -r requirements_tflite.txt

# Download TF Lite model with metadata.
FILE=${DATA_DIR}/efficientnet_lite0.tflite
if [ ! -f "$FILE" ]; then
  curl \
    -L 'https://tfhub.dev/tensorflow/lite-model/efficientnet/lite0/uint8/2?lite-format=tflite' \
    -o ${FILE}
fi

FILE=${DATA_DIR}/efficientnet_lite0_edgetpu.tflite
if [ ! -f "$FILE" ]; then
  curl \
    -L 'https://storage.googleapis.com/download.tensorflow.org/models/tflite/edgetpu/efficientnet-edgetpu-M_quant_edgetpu.tflite' \
    -o ${FILE}
fi
echo -e "Downloaded files are in ${DATA_DIR}"