#!/bin/bash

if [ $# -eq 0 ]; then
  DATA_DIR="./"
else
  DATA_DIR="$1"
fi

# Install Python dependencies
python3 -m pip install -r requirements_pypi.txt
python3 -m pip install -r requirements_tflite.txt

# Download TF Lite models with metadata.
FILE=${DATA_DIR}/deeplabv3.tflite
if [ ! -f "$FILE" ]; then
  curl \
    -L 'https://tfhub.dev/tensorflow/lite-model/deeplabv3/1/metadata/2?lite-format=tflite' \
    -o ${FILE}
fi

FILE=${DATA_DIR}/deeplabv3_edgetpu.tflite
if [ ! -f "$FILE" ]; then
  curl \
    -L 'https://storage.googleapis.com/download.tensorflow.org/models/tflite/edgetpu/deeplabv3_mnv2_dm05_pascal_quant_edgetpu.tflite' \
    -o ${FILE}
fi

echo -e "Downloaded files are in ${DATA_DIR}"