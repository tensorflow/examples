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

#include "lib/tflite_cbr_builder.h"

#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "lib/tflite_builder.h"
#include "tensorflow_lite_support/cc/port/status_macros.h"

namespace tflite {
namespace examples {
namespace cbr {

namespace {

using ::flatbuffers::FlatBufferBuilder;
using ::tflite::task::vision::FeatureVector;

FeatureVector Normalize(const FeatureVector& fv) {
  float norm = 0.0f;
  for (float val : fv.value_float()) norm += val * val;
  if (norm == 0.0f) return fv;  // Can't normalize 0 vector.
  FeatureVector normalized;
  float inv_norm = 1.0f / std::sqrt(norm);
  for (float val : fv.value_float()) {
    normalized.add_value_float(val * inv_norm);
  }
  return normalized;
}

int AddRetrievalBlock(flatbuffers::FlatBufferBuilder* fb_builder,
                      TfLiteBuilder* tflite_builder,
                      int32_t embedding_output_index,
                      const std::vector<FeatureVector>& embeddings,
                      const std::string& output_tensor_name) {
  const int32_t embedding_dim = embeddings[0].value_float_size();
  const int32_t num_instances = embeddings.size();

  // Add a normalization layer.
  const int32_t norm_tensor_index = tflite_builder->AddTensor(
      "normalization", tflite::TensorType_FLOAT32, {1, embedding_dim});

  // Add a weights tensor for the fully-connected retrieval layer.
  // NOTE: We only support a float embedding layer.
  std::vector<float> weights_vec;
  weights_vec.reserve(num_instances * embedding_dim);
  for (const FeatureVector& embedding : embeddings) {
    FeatureVector normalized_embedding = Normalize(embedding);
    for (float val : normalized_embedding.value_float()) {
      weights_vec.push_back(val);
    }
  }
  const int32_t weight_tensor_index = tflite_builder->AddConstTensor(
      "retrieval", tflite::TensorType_FLOAT32, {num_instances, embedding_dim},
      reinterpret_cast<const uint8_t*>(weights_vec.data()),
      sizeof(float) * weights_vec.size());

  // Add a retrieval result tensor.
  const int32_t instances_tensor_index = tflite_builder->AddTensor(
      output_tensor_name, tflite::TensorType_FLOAT32, {1, num_instances});

  // Add a normalization operation. This operation is necessary if we want the
  // final score be meaningful.
  tflite_builder->AddOperator(
      tflite::BuiltinOperator_L2_NORMALIZATION, {embedding_output_index},
      norm_tensor_index, tflite::BuiltinOptions_L2NormOptions,
      tflite::CreateL2NormOptions(*fb_builder,
                                  tflite::ActivationFunctionType_NONE)
          .Union());

  // Add a retrieval operation.
  tflite_builder->AddOperator(
      tflite::BuiltinOperator_FULLY_CONNECTED,
      {norm_tensor_index, weight_tensor_index}, instances_tensor_index,
      tflite::BuiltinOptions_FullyConnectedOptions,
      tflite::CreateFullyConnectedOptions(*fb_builder,
                                          tflite::ActivationFunctionType_NONE)
          .Union());

  return instances_tensor_index;
}

int AddQuantizedRetrievalBlock(flatbuffers::FlatBufferBuilder* fb_builder,
                               TfLiteBuilder* tflite_builder,
                               int32_t embedding_output_index,
                               const std::vector<FeatureVector>& embeddings,
                               const std::string& output_tensor_name) {
  const int32_t embedding_dim = embeddings[0].value_float_size();
  const int32_t num_instances = embeddings.size();

  const int32_t float_embedding_tensor_index = tflite_builder->AddTensor(
      "dequantized_embedding", tflite::TensorType_FLOAT32, {1, embedding_dim});
  const int32_t normalized_embedding_tensor_index = tflite_builder->AddTensor(
      "normalized_embedding", tflite::TensorType_FLOAT32, {1, embedding_dim});
  tflite::QuantizationParametersT embedding_qparams;
  embedding_qparams.min.push_back(-1.0f);
  embedding_qparams.max.push_back(1.0f);
  embedding_qparams.zero_point.push_back(0);
  // See the retrieval layer's quantization parameter setting below.
  embedding_qparams.scale.push_back(1.0f / 128.0f);
  const int32_t nq_embedding_tensor_index = tflite_builder->AddQuantizedTensor(
      "normalized_quantized_embd", tflite::TensorType_INT8, {1, embedding_dim},
      embedding_qparams);

  // Add a dequantization operation.
  // Note that the normalization and matmul operations don't honor the
  // zero-point in a UINT8 tensor (i.e., it doesn't deal with negative values).
  // Therefore, we do: UINT8 --dequantize--> FLOAT32 --normalize-->
  //                   FLOAT32 --quantize--> INT8 --matmul(retrieval)-->
  //                   INT8 --dequantize--> FLOAT32
  tflite_builder->AddOperator(
      tflite::BuiltinOperator_DEQUANTIZE, {embedding_output_index},
      float_embedding_tensor_index, tflite::BuiltinOptions_NONE, 0);

  // Add a normalization operation. This operation is necessary if we want the
  // final score be meaningful.
  tflite_builder->AddOperator(
      tflite::BuiltinOperator_L2_NORMALIZATION, {float_embedding_tensor_index},
      normalized_embedding_tensor_index, tflite::BuiltinOptions_L2NormOptions,
      tflite::CreateL2NormOptions(*fb_builder,
                                  tflite::ActivationFunctionType_NONE)
          .Union());

  // Add a (re-)quantization operation.
  tflite_builder->AddOperator(
      tflite::BuiltinOperator_QUANTIZE, {normalized_embedding_tensor_index},
      nq_embedding_tensor_index, tflite::BuiltinOptions_NONE, 0);

  // Add a weights tensor for the fully-connected retrieval layer.
  // NOTE: We only support a float embedding layer.
  std::vector<int8_t> weights_vec;
  weights_vec.reserve(num_instances * embedding_dim);

  // Most efficient quantization scale would be something like 1/(sqrt(d)*64.0)
  // (calculated with average norm) when d is the embedding dimension. However,
  // if the scale is too big, the matmul (fully-connected) operation would
  // overflow (become larger than 128) and the result would be less accurate
  // for similar embeddings. Therefore, we use the standard scale which is
  // 1/128 for now.
  float scale = 1.0f / 128.0f;
  float scale_inv = 1.0f / scale;
  float zero_point = 0;
  for (const FeatureVector& embedding : embeddings) {
    FeatureVector normalized_embedding = Normalize(embedding);
    for (float val : normalized_embedding.value_float()) {
      float quantized_val = std::round(val * scale_inv) + zero_point;
      if (quantized_val < -128) quantized_val = -128;
      if (quantized_val > 127) quantized_val = 127;
      weights_vec.push_back(static_cast<int8_t>(quantized_val));
    }
  }
  tflite::QuantizationParametersT weight_qparams;
  weight_qparams.min.push_back(-1.0f);
  weight_qparams.max.push_back(1.0f);
  weight_qparams.zero_point.push_back(zero_point);
  weight_qparams.scale.push_back(scale);
  const int32_t weight_tensor_index = tflite_builder->AddQuantizedConstTensor(
      "retrieval", tflite::TensorType_INT8, {num_instances, embedding_dim},
      reinterpret_cast<const uint8_t*>(weights_vec.data()), weights_vec.size(),
      weight_qparams);

  // Add a quantized retrieval result tensor.
  tflite::QuantizationParametersT instances_qparams;
  instances_qparams.min.push_back(-1.0f);
  instances_qparams.max.push_back(1.0f);
  instances_qparams.zero_point.push_back(0);
  instances_qparams.scale.push_back(scale);
  const int32_t quantized_instances_tensor_index =
      tflite_builder->AddQuantizedTensor("quantized_instances",
                                         tflite::TensorType_INT8,
                                         {1, num_instances}, instances_qparams);

  // Add a retrieval operation.
  tflite_builder->AddOperator(
      tflite::BuiltinOperator_FULLY_CONNECTED,
      {nq_embedding_tensor_index, weight_tensor_index},
      quantized_instances_tensor_index,
      tflite::BuiltinOptions_FullyConnectedOptions,
      tflite::CreateFullyConnectedOptions(*fb_builder,
                                          tflite::ActivationFunctionType_NONE)
          .Union());

  // Add a float retrieval result tensor.
  // BEGIN GOOGLE-INTERNAL
  // TODO(187715553): this is a temporary solution. Converting the result back
  // to FLOAT32 is not an optimal solution, but we do that because the
  // aggregation block and the final output does not support INT8. We will need
  // to find a better way to support INT8 in the later stage.
  // END GOOGLE-INTERNAL
  const int32_t instances_tensor_index = tflite_builder->AddTensor(
      output_tensor_name, tflite::TensorType_FLOAT32, {1, num_instances});

  // Add a dequantization operation.
  tflite_builder->AddOperator(
      tflite::BuiltinOperator_DEQUANTIZE, {quantized_instances_tensor_index},
      instances_tensor_index, tflite::BuiltinOptions_NONE, 0);

  return instances_tensor_index;
}

int AddAggregationBlock(flatbuffers::FlatBufferBuilder* fb_builder,
                        TfLiteBuilder* tflite_builder, int num_instances,
                        int instances_tensor_index,
                        const std::vector<std::vector<int32_t>>& classes,
                        const std::string& output_tensor_name) {
  int32_t num_classes = classes.size();

  // Add a flattened retrieval result tensor.
  std::vector<int32_t> flat_instances_shape = {num_instances, 1};
  const int flat_instances_index = tflite_builder->AddTensor(
      "flat_instances", tflite::TensorType_FLOAT32, flat_instances_shape);

  // Add a flattening operation.
  tflite_builder->AddOperator(
      tflite::BuiltinOperator_RESHAPE, {instances_tensor_index},
      flat_instances_index, tflite::BuiltinOptions_ReshapeOptions,
      tflite::CreateReshapeOptions(
          *fb_builder, fb_builder->CreateVector(flat_instances_shape))
          .Union());

  // Add a class aggregation axis tensor.
  std::vector<int32_t> aggregation_axis_vec = {0};
  const int aggregation_axis_tensor_index = tflite_builder->AddConstTensor(
      "aggregation_axis", tflite::TensorType_INT32, {1},
      reinterpret_cast<const uint8_t*>(aggregation_axis_vec.data()),
      sizeof(int32_t) * aggregation_axis_vec.size());

  // Add an aggregation block per class.
  std::vector<int32_t> class_aggregation_indices;
  for (int i = 0; i < classes.size(); ++i) {
    const std::string index_str = std::to_string(i);
    const std::vector<int32_t>& instances = classes[i];
    int32_t num_instances = instances.size();

    // Add a selection tensor.
    const int selection_tensor_index = tflite_builder->AddConstTensor(
        "selection" + index_str, tflite::TensorType_INT32, {num_instances},
        reinterpret_cast<const uint8_t*>(instances.data()),
        sizeof(int32_t) * num_instances);

    // Add a class instances tensor.
    const int class_instances_tensor_index = tflite_builder->AddTensor(
        "class_instances" + index_str, tflite::TensorType_FLOAT32,
        {1, num_instances});

    // Add a class aggregation tensor.
    int class_aggregation_tensor_index = tflite_builder->AddTensor(
        "class_aggregation" + index_str, tflite::TensorType_FLOAT32, {1});

    // Add a class selection operation.
    tflite_builder->AddOperator(tflite::BuiltinOperator_EMBEDDING_LOOKUP,
                                {selection_tensor_index, flat_instances_index},
                                class_instances_tensor_index,
                                tflite::BuiltinOptions_NONE, 0);

    // Add a class aggregation operation.
    tflite_builder->AddOperator(
        tflite::BuiltinOperator_REDUCE_MAX,
        {class_instances_tensor_index, aggregation_axis_tensor_index},
        class_aggregation_tensor_index, tflite::BuiltinOptions_NONE, 0);

    class_aggregation_indices.push_back(class_aggregation_tensor_index);
  }

  // Add the concatenated classes tensor (with dimension {num_classes}).
  const int flat_classes_index = tflite_builder->AddTensor(
      "flat_classes", tflite::TensorType_FLOAT32, {num_classes});

  // Add the final classes tensor (with dimension {1, num_classes}).
  std::vector<int32_t> classes_shape = {1, num_classes};
  const int classes_index = tflite_builder->AddTensor(
      output_tensor_name, tflite::TensorType_FLOAT32, classes_shape);

  // Add a result concatenation operation.
  tflite_builder->AddOperator(tflite::BuiltinOperator_CONCATENATION,
                              class_aggregation_indices, flat_classes_index,
                              tflite::BuiltinOptions_NONE, 0);

  // Add a flattening operation.
  tflite_builder->AddOperator(
      tflite::BuiltinOperator_RESHAPE, {flat_classes_index}, classes_index,
      tflite::BuiltinOptions_ReshapeOptions,
      tflite::CreateReshapeOptions(*fb_builder,
                                   fb_builder->CreateVector(classes_shape))
          .Union());

  return classes_index;
}

}  // namespace

tflite::support::StatusOr<std::vector<std::string>>
TfLiteCbRBuilder::BuildCbRModel(const tflite::Model& model,
                                const std::vector<FeatureVector>& embeddings,
                                const std::vector<std::string>& labels,
                                flatbuffers::FlatBufferBuilder* builder) {
  // Check input arguments.
  if (builder == nullptr) {
    return absl::InvalidArgumentError("FlatBufferBuilder not provided");
  }
  if (model.subgraphs() == nullptr) {
    return absl::InvalidArgumentError("Provided model is empty");
  }
  if (model.subgraphs()->size() != 1) {
    return absl::InvalidArgumentError(
        absl::StrFormat("The model is required to have a single subgraph: %d",
                        model.subgraphs()->size()));
  }
  if (embeddings.empty()) {
    return absl::InvalidArgumentError("Provided embeddings is empty");
  }
  tflite::SubGraphT subgraph_t;
  (*model.subgraphs())[0]->UnPackTo(&subgraph_t);
  if (subgraph_t.outputs.empty()) {
    return absl::InvalidArgumentError("Invalid output size of the model");
  }
  int32_t embedding_output = subgraph_t.outputs[0];

  ASSIGN_OR_RETURN(auto tflite_builder, TfLiteBuilder::New(model, builder));

  const std::string kInstancesTensorName = "instances";
  const std::string kClassesTensorName = "classes";
  int output_index;

  // Check if the embedding is quantized.
  tflite::TensorT* embedding_tensor_t =
      subgraph_t.tensors[embedding_output].get();
  if (!embedding_tensor_t) {
    return absl::InvalidArgumentError("Empty embedding tensor");
  }
  if (embedding_tensor_t->type == tflite::TensorType_FLOAT32) {
    output_index = AddRetrievalBlock(
        builder, tflite_builder.get(), embedding_output, embeddings,
        labels.empty() ? kClassesTensorName : kInstancesTensorName);
  } else {
    // If the embedding is quantized, add a quantized retrieval block which is
    // 4x smaller. Note that the embeddings will be normalized -- which means
    // that the quantization parameter will change.
    output_index = AddQuantizedRetrievalBlock(
        builder, tflite_builder.get(), embedding_output, embeddings,
        labels.empty() ? kClassesTensorName : kInstancesTensorName);
  }

  std::vector<std::string> class_labels;
  if (!labels.empty()) {
    if (embeddings.size() != labels.size()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Labels are not consistent with the embeddings: ", embeddings.size(),
          " vs ", labels.size()));
    }
    absl::flat_hash_map<std::string, int> label_to_class_id;
    std::vector<std::vector<int32_t>> classes;
    int num_classes = 0;
    for (int i = 0; i < labels.size(); ++i) {
      const std::string& label = labels[i];
      int class_id;
      auto itr = label_to_class_id.find(label);
      if (itr == label_to_class_id.end()) {
        class_id = num_classes++;
        label_to_class_id[label] = class_id;
        class_labels.push_back(label);
        classes.push_back({});
      } else {
        class_id = itr->second;
      }
      classes[class_id].push_back(i);
    }

    // Add aggregation block and update output index
    output_index =
        AddAggregationBlock(builder, tflite_builder.get(), embeddings.size(),
                            output_index, classes, kClassesTensorName);
  }

  // Finalize.
  tflite_builder->Build(
      subgraph_t.inputs, output_index, subgraph_t.name, model.version(),
      (model.description() ? model.description()->str() : ""));

  return class_labels;
}

}  // namespace cbr
}  // namespace examples
}  // namespace tflite
