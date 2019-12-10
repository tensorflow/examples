#!/usr/bin/env bash

# Generate the model used in the :transfer_api module tests.

set -e

virtualenv --python python3 .temp_venv
source .temp_venv/bin/activate

pip install ../../converter

mkdir -p ./src/androidTest/assets/

tflite-transfer-convert \
  --base_mobilenetv2 \
  --base_quantize \
  --head_softmax \
  --num_classes 5 \
  --optimizer=sgd \
  --sgd_learning_rate 0.0003 \
  --out_model_dir ./src/androidTest/assets/model

python ./generate_test_resources.py

rm -r .temp_venv
