# Download TF Lite models from the internet if it does not exist.
MODEL_FOLDER=$(dirname "$0")/StyleTransfer/model
PREDICT_INT8_MODEL=${MODEL_FOLDER}/style_predict_quantized_256.tflite
TRANSFORM_INT8_MODEL=${MODEL_FOLDER}/style_transfer_quantized_384.tflite
PREDICT_F16_MODEL=${MODEL_FOLDER}/style_predict_f16_256.tflite
TRANSFORM_F16_MODEL=${MODEL_FOLDER}/style_transfer_f16_384.tflite

if [[ -f "$PREDICT_INT8_MODEL" &&
      -f "$TRANSFORM_INT8_MODEL" &&
      -f "$PREDICT_F16_MODEL" &&
      -f "$TRANSFORM_F16_MODEL" ]]; then
  echo "INFO: TF Lite model already exists. Skip downloading and use the local model."
else
  mkdir -p ${MODEL_FOLDER}
  curl -o ${PREDICT_INT8_MODEL} -L https://storage.googleapis.com/download.tensorflow.org/models/tflite/task_library/style_transfer/ios/magenta_arbitrary-image-stylization-v1-256_int8_prediction_1.tflite
  curl -o ${TRANSFORM_INT8_MODEL} -L https://storage.googleapis.com/download.tensorflow.org/models/tflite/task_library/style_transfer/ios/magenta_arbitrary-image-stylization-v1-256_int8_transfer_1.tflite
  curl -o ${PREDICT_F16_MODEL} -L https://storage.googleapis.com/download.tensorflow.org/models/tflite/task_library/style_transfer/ios/magenta_arbitrary-image-stylization-v1-256_fp16_prediction_1.tflite
  curl -o ${TRANSFORM_F16_MODEL} -L https://storage.googleapis.com/download.tensorflow.org/models/tflite/task_library/style_transfer/ios/magenta_arbitrary-image-stylization-v1-256_fp16_transfer_1.tflite
  echo "INFO: Downloaded TensorFlow Lite model to $MODEL_FOLDER ."
fi
