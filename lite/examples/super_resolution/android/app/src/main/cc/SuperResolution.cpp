/*
 * Copyright 2020 The TensorFlow Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "SuperResolution.h"

#include <android/log.h>
#include <math.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace tflite {
namespace examples {
namespace superresolution {

// TODO: make it changeable in the UI
constexpr int kThreadNum = 4;

SuperResolution::SuperResolution(const void* model_data, size_t model_size,
                                 bool use_gpu) {
  // Load the model
  model_ = TfLiteModelCreate(model_data, model_size);
  if (!model_) {
    LOGE("Failed to create TFLite model");
    return;
  }

  // Create the interpreter options
  options_ = TfLiteInterpreterOptionsCreate();

  // Choose CPU or GPU
  if (use_gpu) {
    delegate_ = TfLiteGpuDelegateV2Create(/*default options=*/nullptr);
    TfLiteInterpreterOptionsAddDelegate(options_, delegate_);
  } else {
    TfLiteInterpreterOptionsSetNumThreads(options_, kThreadNum);
  }

  // Create the interpreter
  interpreter_ = TfLiteInterpreterCreate(model_, options_);
  if (!interpreter_) {
    LOGE("Failed to create TFLite interpreter");
    return;
  }
}

SuperResolution::~SuperResolution() {
  // Dispose of the model and interpreter objects
  if (interpreter_) {
    TfLiteInterpreterDelete(interpreter_);
  }
  if (delegate_) {
    TfLiteGpuDelegateV2Delete(delegate_);
  }
  if (options_) {
    TfLiteInterpreterOptionsDelete(options_);
  }
  if (model_) {
    TfLiteModelDelete(model_);
  }
}

bool SuperResolution::IsInterpreterCreated() {
  if (!interpreter_) {
    return false;
  } else {
    return true;
  }
}

std::unique_ptr<int[]> SuperResolution::DoSuperResolution(int* lr_img_rgb) {
  // Allocate tensors and populate the input tensor data
  TfLiteStatus status = TfLiteInterpreterAllocateTensors(interpreter_);
  if (status != kTfLiteOk) {
    LOGE("Something went wrong when allocating tensors");
    return nullptr;
  }

  TfLiteTensor* input_tensor =
      TfLiteInterpreterGetInputTensor(interpreter_, 0);

  // Extract RGB values from each pixel
  float input_buffer[kNumberOfInputPixels * kImageChannels];
  for (int i = 0, j = 0; i < kNumberOfInputPixels; i++) {
    // Alpha is ignored
    input_buffer[j++] = static_cast<float>((lr_img_rgb[i] >> 16) & 0xff);
    input_buffer[j++] = static_cast<float>((lr_img_rgb[i] >> 8) & 0xff);
    input_buffer[j++] = static_cast<float>((lr_img_rgb[i]) & 0xff);
  }

  // Feed input into model
  status = TfLiteTensorCopyFromBuffer(
      input_tensor, input_buffer,
      kNumberOfInputPixels * kImageChannels * sizeof(float));
  if (status != kTfLiteOk) {
    LOGE("Something went wrong when copying input buffer to input tensor");
    return nullptr;
  }

  // Run the interpreter
  status = TfLiteInterpreterInvoke(interpreter_);
  if (status != kTfLiteOk) {
    LOGE("Something went wrong when running the TFLite model");
    return nullptr;
  }

  // Extract the output tensor data
  const TfLiteTensor* output_tensor =
      TfLiteInterpreterGetOutputTensor(interpreter_, 0);
  float output_buffer[kNumberOfOutputPixels * kImageChannels];
  status = TfLiteTensorCopyToBuffer(
      output_tensor, output_buffer,
      kNumberOfOutputPixels * kImageChannels * sizeof(float));
  if (status != kTfLiteOk) {
    LOGE("Something went wrong when copying output tensor to output buffer");
    return nullptr;
  }

  // Postprocess the output from TFLite
  int clipped_output[kImageChannels];
  auto rgb_colors = std::make_unique<int[]>(kNumberOfOutputPixels);
  for (int i = 0; i < kNumberOfOutputPixels; i++) {
    for (int j = 0; j < kImageChannels; j++) {
      clipped_output[j] = std::max<float>(
          0, std::min<float>(255, output_buffer[i * kImageChannels + j]));
    }
    // When we have RGB values, we pack them into a single pixel.
    // Alpha is set to 255.
    rgb_colors[i] = (255u & 0xff) << 24 | (clipped_output[0] & 0xff) << 16 |
                    (clipped_output[1] & 0xff) << 8 |
                    (clipped_output[2] & 0xff);
  }

  return rgb_colors;
}

}  // namespace superresolution
}  // namespace examples
}  // namespace tflite
