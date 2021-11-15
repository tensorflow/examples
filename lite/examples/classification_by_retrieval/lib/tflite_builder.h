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

#ifndef TENSORFLOW_LITE_EXAMPLES_CLASSIFICATION_BY_RETRIEVAL_LIB_TFLITE_BUILDER_H_
#define TENSORFLOW_LITE_EXAMPLES_CLASSIFICATION_BY_RETRIEVAL_LIB_TFLITE_BUILDER_H_

#include <memory>
#include <string>

#include "absl/types/optional.h"
#include "flatbuffers/flatbuffers.h"
#include "tensorflow/lite/model.h"
#include "tensorflow_lite_support/cc/port/statusor.h"

namespace tflite {
namespace examples {
namespace cbr {

class TfLiteBuilder {
 public:
  // Creates a new instance of TfLiteBuilderImpl.
  // The caller keeps the ownership of `builder`.
  static tflite::support::StatusOr<std::unique_ptr<TfLiteBuilder>> New(
      ::flatbuffers::FlatBufferBuilder* fbb);

  // Creates a new instance of TfLiteBuilder and initializes it by copying
  // all the data in `model`. The caller keeps the ownership of `builder`.
  static tflite::support::StatusOr<std::unique_ptr<TfLiteBuilder>> New(
      const Model& model, ::flatbuffers::FlatBufferBuilder* fbb);

  virtual ~TfLiteBuilder() = default;

  // Add a tensor to the model and returns the index.
  virtual int AddTensor(const std::string& name, TensorType type,
                        const std::vector<int32_t>& shape) = 0;

  // AddTensor with a quantization parameter.
  virtual int AddQuantizedTensor(
      const std::string& name, TensorType type,
      const std::vector<int32_t>& shape,
      const QuantizationParametersT& quantization_param) = 0;

  // Add a tensor with a constant value and returns the index.
  virtual int AddConstTensor(const std::string& name, TensorType type,
                             const std::vector<int32_t>& shape,
                             const uint8_t* data, size_t size) = 0;

  // AddConstTensor with a quantization parameter.
  virtual int AddQuantizedConstTensor(
      const std::string& name, TensorType type,
      const std::vector<int32_t>& shape, const uint8_t* data, size_t size,
      const QuantizationParametersT& quantization_opt) = 0;

  // Add an operator to the model.
  virtual void AddOperator(BuiltinOperator op_code,
                           const std::vector<int>& input_tensor_indices,
                           int output_tensor_index, BuiltinOptions option_code,
                           const flatbuffers::Offset<void>& options) = 0;

  virtual void Build(const std::vector<int32_t>& inputs,
                     int32_t output_tensor_index, const std::string& name,
                     uint32_t schema_version,
                     const std::string& description) = 0;
};

}  // namespace cbr
}  // namespace examples
}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXAMPLES_CLASSIFICATION_BY_RETRIEVAL_LIB_TFLITE_BUILDER_H_
