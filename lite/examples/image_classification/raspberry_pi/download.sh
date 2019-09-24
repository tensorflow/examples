#!/bin/bash

if [ $# -eq 0 ]; then
  DATA_DIR="/tmp"
else
  DATA_DIR="$1"
fi

# Install required packages
python3 -m pip install -r requirements.txt

# Get TF Lite model and labels
curl -O https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v1_1.0_224_quant_and_labels.zip

unzip mobilenet_v1_1.0_224_quant_and_labels.zip -d ${DATA_DIR}

rm mobilenet_v1_1.0_224_quant_and_labels.zip

# Get version compiled for Edge TPU
curl https://dl.google.com/coral/canned_models/mobilenet_v1_1.0_224_quant_edgetpu.tflite \
-o ${DATA_DIR}/mobilenet_v1_1.0_224_quant_edgetpu.tflite

echo -e "Downloaded files are in ${DATA_DIR}"
