// Copyright 2021 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "lib/labeled_image_helper.h"

#include <string>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "lib/model_builder.h"
#include "tensorflow_lite_support/cc/port/status_macros.h"
#include "tensorflow_lite_support/cc/port/statusor.h"
#include "tensorflow_lite_support/cc/task/vision/core/frame_buffer.h"
#include "tensorflow_lite_support/cc/task/vision/utils/frame_buffer_common_utils.h"
#include "tensorflow_lite_support/examples/task/vision/desktop/utils/image_utils.h"

namespace tflite {
namespace examples {
namespace cbr {

namespace {

using ::tflite::task::vision::CreateFromRgbaRawBuffer;
using ::tflite::task::vision::CreateFromRgbRawBuffer;
using ::tflite::task::vision::DecodeImageFromFile;
using ::tflite::task::vision::FrameBuffer;
using ::tflite::task::vision::ImageData;
using ::tflite::task::vision::ImageDataFree;

}  // namespace

tflite::support::StatusOr<std::unique_ptr<FrameBuffer>>
BuildFrameBufferFromImageData(const ImageData& image) {
  std::unique_ptr<FrameBuffer> frame_buffer;
  if (image.channels == 3) {
    return CreateFromRgbRawBuffer(image.pixel_data,
                                  {image.width, image.height});
  } else if (image.channels == 4) {
    return CreateFromRgbaRawBuffer(image.pixel_data,
                                   {image.width, image.height});
  }
  return absl::InvalidArgumentError(
      absl::StrCat("Expected image with 3 (RGB) or 4 (RGBA) channels, found ",
                   image.channels));
}

absl::Status AddLabeledImageFromPath(ModelBuilder* model_builder,
                                     const std::string& label,
                                     const std::string& image_file_path) {
  // Decode image and load into a FrameBuffer.
  ASSIGN_OR_RETURN(ImageData image_data, DecodeImageFromFile(image_file_path));
  ASSIGN_OR_RETURN(std::unique_ptr<FrameBuffer> frame_buffer,
                   BuildFrameBufferFromImageData(image_data));
  // Add to model builder.
  RETURN_IF_ERROR(model_builder->AddLabeledImage(label, *frame_buffer));
  // Cleanup.
  ImageDataFree(&image_data);
  return absl::OkStatus();
}

}  // namespace cbr
}  // namespace examples
}  // namespace tflite
