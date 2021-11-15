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

#include "lib/tflite_builder.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/types/optional.h"
#include "flatbuffers/flatbuffers.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow_lite_support/cc/port/status_macros.h"
#include "tensorflow_lite_support/cc/port/statusor.h"

namespace tflite {
namespace examples {
namespace cbr {

namespace {

using ::flatbuffers::FlatBufferBuilder;
using ::flatbuffers::Offset;
using ::flatbuffers::Vector;

// Re-implementation of tflite::GetBuiltinCode.
BuiltinOperator GetBuiltinCode(const OperatorCodeT& op_code) {
  return std::max(op_code.builtin_code, static_cast<BuiltinOperator>(
                                            op_code.deprecated_builtin_code));
}

Offset<QuantizationParameters> CreateQuantizationParameters(
    FlatBufferBuilder* fbb, const QuantizationParametersT& params) {
  return ::tflite::CreateQuantizationParameters(
      *fbb, params.min.empty() ? 0 : fbb->CreateVector(params.min),
      params.max.empty() ? 0 : fbb->CreateVector(params.max),
      params.scale.empty() ? 0 : fbb->CreateVector(params.scale),
      params.zero_point.empty() ? 0 : fbb->CreateVector(params.zero_point));
}

// A convenience class for adding layers to a TF Lite model.
class TfLiteBuilderImpl : public TfLiteBuilder {
 public:
  // Creates a new instance of TfLiteBuilderImpl.
  // The caller keeps the ownership of `fbb`.
  static tflite::support::StatusOr<std::unique_ptr<TfLiteBuilderImpl>> New(
      FlatBufferBuilder* fbb);

  // Creates a new instance of TfLiteBuilderImpl and initializes it by copying
  // all the data in `model`. The caller keeps the ownership of `fbb`.
  static tflite::support::StatusOr<std::unique_ptr<TfLiteBuilderImpl>> New(
      const tflite::Model& model, FlatBufferBuilder* fbb);

  int AddTensor(const std::string& name, TensorType type,
                const std::vector<int32_t>& shape) override {
    return AddTensor(name, type, shape, absl::nullopt);
  }

  int AddQuantizedTensor(
      const std::string& name, TensorType type,
      const std::vector<int32_t>& shape,
      const QuantizationParametersT& quantization_param) override {
    return AddTensor(name, type, shape, quantization_param);
  }

  int AddConstTensor(const std::string& name, TensorType type,
                     const std::vector<int32_t>& shape, const uint8_t* data,
                     size_t size) override {
    return AddConstTensor(name, type, shape, data, size, absl::nullopt);
  }

  int AddQuantizedConstTensor(
      const std::string& name, TensorType type,
      const std::vector<int32_t>& shape, const uint8_t* data, size_t size,
      const QuantizationParametersT& quantization_param) override {
    return AddConstTensor(name, type, shape, data, size, quantization_param);
  }

  void AddOperator(BuiltinOperator op_code,
                   const std::vector<int>& input_tensor_indices,
                   int output_tensor_index, BuiltinOptions option_code,
                   const flatbuffers::Offset<void>& options) override;

  void Build(const std::vector<int32_t>& inputs, int32_t output_tensor_index,
             const std::string& name, uint32_t version,
             const std::string& description) override;

 private:
  // Caller keeps the ownership of `fbb`.
  explicit TfLiteBuilderImpl(FlatBufferBuilder* fbb) : fbb_(fbb) {}

  // Clones all model buffers.
  absl::Status CloneBuffers(const Model& model);
  // Clones all model tensors, assuming the model has only one subgraph. It also
  // assumes that the buffers were cloned (only once) using the above
  // CloneBuffers().
  absl::Status CloneTensors(const Model& model);
  // Clones all model operator codes. No deduping.
  absl::Status CloneOperatorCodes(const Model& model);
  // Clones all model operators. It assumes that the operator codes were cloned
  // (only once) using the above CloneOperatorCodes().
  absl::Status CloneOperators(const Model& model);
  // Clones all metadata. In fact, most metadata (model name, description, etc)
  // needs to be modified but for now, we clone everything for easiness of
  // implementation.
  absl::Status CloneMetadata(const Model& model);

  int AddTensor(
      const std::string& name, TensorType type,
      const std::vector<int32_t>& shape,
      const absl::optional<QuantizationParametersT>& quantization_opt);

  int AddConstTensor(
      const std::string& name, TensorType type,
      const std::vector<int32_t>& shape, const uint8_t* data, size_t size,
      const absl::optional<QuantizationParametersT>& quantization_opt);

  FlatBufferBuilder* fbb_;
  std::vector<Offset<Buffer>> buffer_vector_;
  std::vector<Offset<Tensor>> tensor_vector_;
  std::vector<Offset<OperatorCode>> opcode_vector_;
  std::vector<Offset<Operator>> op_vector_;
  std::vector<Offset<Metadata>> metadata_;
  absl::flat_hash_map<BuiltinOperator, int> op_index_;
};

/*static*/
tflite::support::StatusOr<std::unique_ptr<TfLiteBuilderImpl>>
TfLiteBuilderImpl::New(FlatBufferBuilder* fbb) {
  if (!fbb) return absl::InvalidArgumentError("fbb is required");
  return absl::WrapUnique(new TfLiteBuilderImpl(fbb));
}

/*static*/
tflite::support::StatusOr<std::unique_ptr<TfLiteBuilderImpl>>
TfLiteBuilderImpl::New(const Model& model, FlatBufferBuilder* fbb) {
  ASSIGN_OR_RETURN(auto tflite_builder, TfLiteBuilderImpl::New(fbb));
  RETURN_IF_ERROR(tflite_builder->CloneBuffers(model));
  RETURN_IF_ERROR(tflite_builder->CloneTensors(model));
  RETURN_IF_ERROR(tflite_builder->CloneOperatorCodes(model));
  RETURN_IF_ERROR(tflite_builder->CloneOperators(model));
  RETURN_IF_ERROR(tflite_builder->CloneMetadata(model));
  return std::move(tflite_builder);
}

absl::Status TfLiteBuilderImpl::CloneBuffers(const Model& model) {
  for (int i = 0; i < model.buffers()->size(); ++i) {
    auto* buffer = model.buffers()->Get(i);
    if (buffer->data() == nullptr) {
      // Many transient tensors don't have data in the flatbuffer. Their
      // buffers will be allocated by the interpreter at run-time.
      buffer_vector_.push_back(CreateBuffer(*fbb_));
    } else {
      BufferT buffer_t;
      buffer->UnPackTo(&buffer_t);
      Offset<Vector<uint8_t>> data_buffer =
          fbb_->CreateVector(buffer_t.data.data(), buffer_t.data.size());
      buffer_vector_.push_back(CreateBuffer(*fbb_, data_buffer));
    }
  }
  return absl::OkStatus();
}

absl::Status TfLiteBuilderImpl::CloneTensors(const Model& model) {
  const auto* subgraphs = model.subgraphs();
  const auto* tensors = subgraphs->Get(0)->tensors();
  for (int i = 0; i < tensors->size(); ++i) {
    const auto* tensor = tensors->Get(i);
    if (!tensor) return absl::InvalidArgumentError("null tensor provided");
    TensorT tensor_t;
    tensor->UnPackTo(&tensor_t);
    tensor_vector_.push_back(CreateTensor(
        *fbb_, fbb_->CreateVector(tensor_t.shape), tensor_t.type,
        tensor_t.buffer, fbb_->CreateString(tensor_t.name),
        tensor_t.quantization
            ? CreateQuantizationParameters(fbb_, *tensor_t.quantization)
            : 0));
  }
  return absl::OkStatus();
}

absl::Status TfLiteBuilderImpl::CloneOperatorCodes(const Model& model) {
  for (int i = 0; i < model.operator_codes()->size(); ++i) {
    const auto* opcode = model.operator_codes()->Get(i);
    OperatorCodeT opcode_t;
    opcode->UnPackTo(&opcode_t);
    BuiltinOperator builtin_code = GetBuiltinCode(opcode_t);
    op_index_[builtin_code] = opcode_vector_.size();
    opcode_vector_.push_back(CreateOperatorCode(
        *fbb_, builtin_code,
        opcode->custom_code() ? fbb_->CreateString(opcode_t.custom_code) : 0,
        opcode_t.version));
  }
  return absl::OkStatus();
}

absl::Status TfLiteBuilderImpl::CloneOperators(const Model& model) {
  const auto* ops = model.subgraphs()->Get(0)->operators();
  for (int i = 0; i < ops->size(); ++i) {
    const auto* op = ops->Get(i);
    if (!op->inputs()) return absl::InvalidArgumentError("empty op inputs");
    if (!op->outputs()) return absl::InvalidArgumentError("empty op outputs");

    OperatorT op_t;
    op->UnPackTo(&op_t);

    // Recalculate input and output indices of this operator.
    Offset<Vector<int32_t>> input_index_vector =
        fbb_->CreateVector<int32_t>(op_t.inputs);
    Offset<Vector<int32_t>> output_index_vector =
        fbb_->CreateVector<int32_t>(op_t.outputs);

    const auto builtin_options_type = op_t.builtin_options.type;

    const auto custom_options_format = op_t.custom_options_format;

    op_vector_.push_back(CreateOperator(
        *fbb_, op_t.opcode_index, input_index_vector, output_index_vector,
        builtin_options_type,
        op->builtin_options() ? op_t.builtin_options.Pack(*fbb_) : 0,
        op->custom_options() ? fbb_->CreateVector(op_t.custom_options.data(),
                                                  op_t.custom_options.size())
                             : 0,
        custom_options_format));
  }
  return absl::OkStatus();
}

int TfLiteBuilderImpl::AddTensor(
    const std::string& name, TensorType type, const std::vector<int32_t>& shape,
    const absl::optional<QuantizationParametersT>& quantization_opt) {
  const int buffer_index = buffer_vector_.size();
  buffer_vector_.push_back(CreateBuffer(*fbb_));
  const int tensor_index = tensor_vector_.size();
  tensor_vector_.push_back(CreateTensor(
      *fbb_, fbb_->CreateVector(shape), type, buffer_index,
      fbb_->CreateString(name),
      quantization_opt.has_value()
          ? CreateQuantizationParameters(fbb_, quantization_opt.value())
          : 0));
  return tensor_index;
}

int TfLiteBuilderImpl::AddConstTensor(
    const std::string& name, TensorType type, const std::vector<int32_t>& shape,
    const uint8_t* data, size_t size,
    const absl::optional<QuantizationParametersT>& quantization_opt) {
  const int buffer_index = buffer_vector_.size();
  buffer_vector_.push_back(CreateBuffer(*fbb_, fbb_->CreateVector(data, size)));
  const int tensor_index = tensor_vector_.size();
  tensor_vector_.push_back(CreateTensor(
      *fbb_, fbb_->CreateVector(shape), type, buffer_index,
      fbb_->CreateString(name),
      quantization_opt.has_value()
          ? CreateQuantizationParameters(fbb_, quantization_opt.value())
          : 0));
  return tensor_index;
}

// Appends clones of model metadata to the output metadata vector.
absl::Status TfLiteBuilderImpl::CloneMetadata(const Model& model) {
  const auto* original_metadata_list = model.metadata();
  for (int i = 0; i < original_metadata_list->size(); ++i) {
    const auto* original_metadata = original_metadata_list->Get(i);
    MetadataT original_metadata_t;
    original_metadata->UnPackTo(&original_metadata_t);
    // To do this properly, we need to look at the TFLITE_METADATA field
    // (name = TFLITE_METADATA), parse it, and modify it accordingly. For now,
    // we just clone all the metadata.
    metadata_.push_back(
        CreateMetadata(*fbb_, fbb_->CreateString(original_metadata_t.name),
                       original_metadata_t.buffer));
  }
  return absl::OkStatus();
}

void TfLiteBuilderImpl::AddOperator(
    BuiltinOperator op_code, const std::vector<int32_t>& input_tensor_indices,
    int output_tensor_index, BuiltinOptions option_code,
    const flatbuffers::Offset<void>& options) {
  auto iter = op_index_.find(op_code);
  int index;
  if (iter != op_index_.end()) {
    index = iter->second;
  } else {
    index = opcode_vector_.size();
    opcode_vector_.push_back(CreateOperatorCode(*fbb_, op_code));
  }

  Offset<Vector<int32_t>> input_vector =
      fbb_->CreateVector<int32_t>(input_tensor_indices);
  Offset<Vector<int32_t>> output_vector =
      fbb_->CreateVector<int32_t>({output_tensor_index});
  op_vector_.push_back(CreateOperator(*fbb_, index, input_vector, output_vector,
                                      option_code, options));
}

void TfLiteBuilderImpl::Build(const std::vector<int32_t>& inputs,
                              int32_t output_tensor_index,
                              const std::string& name, uint32_t version,
                              const std::string& description) {
  Offset<Vector<int32_t>> input_vector = fbb_->CreateVector<int32_t>(inputs);
  Offset<Vector<int32_t>> output_vector =
      fbb_->CreateVector<int32_t>({output_tensor_index});
  Offset<Vector<Offset<Tensor>>> tensors = fbb_->CreateVector(tensor_vector_);
  Offset<Vector<Offset<Operator>>> ops = fbb_->CreateVector(op_vector_);
  Offset<SubGraph> subgraph =
      CreateSubGraph(*fbb_, tensors, input_vector, output_vector, ops,
                     name.empty() ? 0 : fbb_->CreateString(name));
  Offset<Vector<Offset<SubGraph>>> subgraphs =
      fbb_->CreateVector<Offset<SubGraph>>({subgraph});

  Offset<Vector<Offset<Buffer>>> buffers = fbb_->CreateVector(buffer_vector_);
  Offset<Vector<Offset<OperatorCode>>> opcodes =
      fbb_->CreateVector(opcode_vector_);

  auto merged_model = tflite::CreateModel(
      *fbb_, version, opcodes, subgraphs,
      description.empty() ? 0 : fbb_->CreateString(description), buffers, 0,
      fbb_->CreateVector(metadata_));

  tflite::FinishModelBuffer(*fbb_, merged_model);
}

}  // namespace

/*static*/
tflite::support::StatusOr<std::unique_ptr<TfLiteBuilder>> TfLiteBuilder::New(
    FlatBufferBuilder* fbb) {
  return TfLiteBuilderImpl::New(fbb);
}

/*static*/
tflite::support::StatusOr<std::unique_ptr<TfLiteBuilder>> TfLiteBuilder::New(
    const Model& model, ::flatbuffers::FlatBufferBuilder* fbb) {
  return TfLiteBuilderImpl::New(model, fbb);
}

}  // namespace cbr
}  // namespace examples
}  // namespace tflite
