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

#ifndef TENSORFLOW_LITE_EXAMPLES_CLASSIFICATION_BY_RETRIEVAL_LIB_TFLITE_CBR_BUILDER_H_
#define TENSORFLOW_LITE_EXAMPLES_CLASSIFICATION_BY_RETRIEVAL_LIB_TFLITE_CBR_BUILDER_H_

#include <string>
#include <vector>

#include "flatbuffers/flatbuffers.h"
#include "tensorflow/lite/model.h"
#include "tensorflow_lite_support/cc/port/statusor.h"
#include "tensorflow_lite_support/cc/task/vision/proto/embeddings_proto_inc.h"

namespace tflite {
namespace examples {
namespace cbr {

// Wrapper class around BuildCbRModel() function for dependency injection.
class TfLiteCbRBuilder {
 public:
  TfLiteCbRBuilder() = default;
  virtual ~TfLiteCbRBuilder() = default;

  // Performs modifications on the provided embedder model to turn it into a
  // classification-by-retrieval model based on the provided list of embeddings.
  // On success, it returns the updated class labels after aggregation or an
  // empty vector if `labels` is empty.
  virtual tflite::support::StatusOr<std::vector<std::string>> BuildCbRModel(
      const ::tflite::Model& model,
      const std::vector<::tflite::task::vision::FeatureVector>& embeddings,
      const std::vector<std::string>& labels,
      ::flatbuffers::FlatBufferBuilder* builder);
};

}  // namespace cbr
}  // namespace examples
}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXAMPLES_CLASSIFICATION_BY_RETRIEVAL_LIB_TFLITE_CBR_BUILDER_H_
