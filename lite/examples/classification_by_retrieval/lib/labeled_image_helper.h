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

#ifndef TENSORFLOW_LITE_EXAMPLES_CLASSIFICATION_BY_RETRIEVAL_LIB_TESTS_LABELED_IMAGE_HELPER_H_
#define TENSORFLOW_LITE_EXAMPLES_CLASSIFICATION_BY_RETRIEVAL_LIB_TESTS_LABELED_IMAGE_HELPER_H_

#include <string>

#include "absl/status/status.h"
#include "lib/model_builder.h"
#include "tensorflow_lite_support/cc/task/vision/core/frame_buffer.h"
#include "tensorflow_lite_support/examples/task/vision/desktop/utils/image_utils.h"

namespace tflite {
namespace examples {
namespace cbr {

// Returns a FrambeBuffer view of `image`.
tflite::support::StatusOr<std::unique_ptr<::tflite::task::vision::FrameBuffer>>
BuildFrameBufferFromImageData(const ::tflite::task::vision::ImageData& image);

// A helper function that reads an image from the given path, converts it to a
// frame buffer and adds it to the model builder with the provided label.
absl::Status AddLabeledImageFromPath(ModelBuilder* model_builder,
                                     const std::string& label,
                                     const std::string& image_file_path);

}  // namespace cbr
}  // namespace examples
}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXAMPLES_CLASSIFICATION_BY_RETRIEVAL_LIB_TESTS_LABELED_IMAGE_HELPER_H_
