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

#ifndef TENSORFLOW_LITE_EXAMPLES_CLASSIFICATION_BY_RETRIEVAL_LIB_MODEL_BUILDER_H_
#define TENSORFLOW_LITE_EXAMPLES_CLASSIFICATION_BY_RETRIEVAL_LIB_MODEL_BUILDER_H_

#include <memory>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "lib/tflite_cbr_builder.h"
#include "tensorflow/lite/model.h"
#include "tensorflow_lite_support/cc/port/statusor.h"
#include "tensorflow_lite_support/cc/task/core/proto/external_file_proto_inc.h"
#include "tensorflow_lite_support/cc/task/vision/core/frame_buffer.h"
#include "tensorflow_lite_support/cc/task/vision/image_embedder.h"
#include "tensorflow_lite_support/cc/task/vision/proto/embeddings_proto_inc.h"
#include "tensorflow_lite_support/cc/task/vision/proto/image_embedder_options_proto_inc.h"

namespace tflite {
namespace examples {
namespace cbr {

// TODO(b/183007585): add tests for this class once complete.
// Encapsulates the logic required to transform an embedder model into a
// classification-by-retrieval model.
class ModelBuilder {
 public:
  // For testing purposes; prefer using CreateFromImageEmbedderOptions.
  explicit ModelBuilder(
      std::unique_ptr<::tflite::task::vision::ImageEmbedder> image_embedder,
      std::unique_ptr<::tflite::FlatBufferModel> model,
      std::unique_ptr<TfLiteCbRBuilder> tflite_cbr_builder =
          absl::make_unique<TfLiteCbRBuilder>());

  // Initializes the ModelBuilder from the provided ImageEmbedderOptions.
  static tflite::support::StatusOr<std::unique_ptr<ModelBuilder>>
  CreateFromImageEmbedderOptions(
      const ::tflite::task::vision::ImageEmbedderOptions& options);

  // Sets the metadata fields associated with the model. This is flushed when
  // `BuildModel()` returns and needs to be called again as needed.
  //
  // `associated_files` is a map of {filename, file contents}.
  // The files are added with the DESCRIPTIONS type.
  void SetMetadata(
      const std::string& name, const std::string& description,
      const std::string& author, const std::string& version,
      const std::string& license,
      const absl::flat_hash_map<std::string, std::string>& associated_files);

  // Extracts the embedding on the provided FrameBuffer and stores it along
  // with the provided label. This method is meant to be called multiple times
  // on each labeled image used to create the classification-by-retrieval model
  // before a final call to `BuildModel()` actually creates and returns the
  // final model.
  absl::Status AddLabeledImage(
      const std::string& label,
      const ::tflite::task::vision::FrameBuffer& frame_buffer);

  // Finalizes the classification-by-retrieval model construction using the
  // feature vectors extracted along the successive (at least two) calls to
  // `AddLabeledImage()`. If less than two labeled images have been added, this
  // function returns an absl::FailedPreconditionError.
  //
  // The returned model includes model metadata (i.e. it's ready for use in the
  // ImageClassifier Task API). It is returned as an ExternalFile proto with the
  // `file_content` field set. It can either be used directly to build an
  // ImageClassifierOption proto and initialize an ImageClassifier, or written
  // to disk as a .tflite file.
  //
  // Except if an error is returned, calling this function also resets the
  // ModelBuilder, so that the process of adding new labeled images and building
  // another model can be repeated.
  tflite::support::StatusOr<::tflite::task::core::ExternalFile> BuildModel();

 private:
  // Populates metadata on the provided model buffer, returns the model with
  // metadata as a string.
  tflite::support::StatusOr<std::string> PopulateMetadata(
      const char* buffer_data, size_t buffer_size);

  // The ImageEmbedder built from options provided at initialization time.
  std::unique_ptr<::tflite::task::vision::ImageEmbedder> image_embedder_;
  // The TfLite embedding model built from options provided at initialization
  // time. This is the identical model that is the used by `image_embedder_`.
  std::unique_ptr<::tflite::FlatBufferModel> model_;
  // The TfLiteCbrBuilder to use for creating the classification-by-retrieval
  // TFLite model.
  std::unique_ptr<TfLiteCbRBuilder> tflite_cbr_builder_;

  // Model details, filled by the last call to `SetMetadata()` and flushed on
  // final call to `BuildModel()`.
  std::string name_;
  std::string description_;
  std::string author_;
  std::string version_;
  std::string license_;
  absl::flat_hash_map<std::string, std::string> associated_files_;
  // The list of currently added labels, filled along successive calls to
  // `AddLabeledImage()` and flushed on final call to `BuildModel()`.
  std::vector<std::string> labels_;
  // The list of currently extracted feature vectors, filled along successive
  // calls to `AddLabeledImage()` and flushed on final call to `BuildModel()`.
  std::vector<::tflite::task::vision::FeatureVector> feature_vectors_;
};

}  // namespace cbr
}  // namespace examples
}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXAMPLES_CLASSIFICATION_BY_RETRIEVAL_LIB_MODEL_BUILDER_H_
