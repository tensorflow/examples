#include "lib/model_builder.h"

#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "flatbuffers/flatbuffers.h"
#include "tensorflow_lite_support/cc/port/status_macros.h"
#include "tensorflow_lite_support/cc/task/vision/proto/embeddings_proto_inc.h"
#include "tensorflow_lite_support/metadata/cc/metadata_populator.h"
#include "tensorflow_lite_support/metadata/metadata_schema_generated.h"

namespace tflite {
namespace examples {
namespace cbr {

namespace {

constexpr char kLabelMapFilename[] = "labelmap.txt";

using ::flatbuffers::FlatBufferBuilder;
using ::tflite::FlatBufferModel;
using ::tflite::metadata::ModelMetadataPopulator;
using ::tflite::task::core::ExternalFile;
using ::tflite::task::vision::EmbeddingResult;
using ::tflite::task::vision::FeatureVector;
using ::tflite::task::vision::FrameBuffer;
using ::tflite::task::vision::ImageEmbedder;
using ::tflite::task::vision::ImageEmbedderOptions;

}  // namespace

ModelBuilder::ModelBuilder(std::unique_ptr<ImageEmbedder> image_embedder,
                           std::unique_ptr<FlatBufferModel> model,
                           std::unique_ptr<TfLiteCbRBuilder> tflite_cbr_builder)
    : image_embedder_(std::move(image_embedder)),
      model_(std::move(model)),
      tflite_cbr_builder_(std::move(tflite_cbr_builder)) {}

/* static */
tflite::support::StatusOr<std::unique_ptr<ModelBuilder>>
ModelBuilder::CreateFromImageEmbedderOptions(
    const ImageEmbedderOptions& options) {
  // Quantization is not supported.
  if (options.quantize()) {
    return absl::UnimplementedError(
        "Scalar quantization through the `quantize` option is not supported.");
  }
  // Build ImageEmbedder.
  ASSIGN_OR_RETURN(std::unique_ptr<ImageEmbedder> image_embedder,
                   ImageEmbedder::CreateFromOptions(options));
  // Multi-head classifiers are not supported either.
  if (image_embedder->GetNumberOfOutputLayers() != 1) {
    return absl::UnimplementedError(
        "Multi-head ImageEmbedder-s are not supported");
  }
  const std::string& model_file =
      options.model_file_with_metadata().file_name();
  std::unique_ptr<FlatBufferModel> model =
      FlatBufferModel::BuildFromFile(model_file.c_str());
  if (model == nullptr) {
    return absl::InvalidArgumentError("Failed to build model from file: " +
                                      model_file);
  }
  return absl::make_unique<ModelBuilder>(std::move(image_embedder),
                                         std::move(model));
}

void ModelBuilder::SetMetadata(
    const std::string& name, const std::string& description,
    const std::string& author, const std::string& version,
    const std::string& license,
    const absl::flat_hash_map<std::string, std::string>& associated_files) {
  name_ = name;
  description_ = description;
  author_ = author;
  version_ = version;
  license_ = license;
  associated_files_ = associated_files;
}

absl::Status ModelBuilder::AddLabeledImage(
    const std::string& label,
    const ::tflite::task::vision::FrameBuffer& frame_buffer) {
  ASSIGN_OR_RETURN(const EmbeddingResult& embedding_result,
                   image_embedder_->Embed(frame_buffer));
  feature_vectors_.emplace_back(
      image_embedder_->GetEmbeddingByIndex(embedding_result, 0)
          .feature_vector());
  labels_.emplace_back(label);
  return absl::OkStatus();
}

tflite::support::StatusOr<std::string> ModelBuilder::PopulateMetadata(
    const char* buffer_data, size_t buffer_size) {
  // Copy metadata from original model.
  tflite::ModelMetadataT model_metadata_t;
  image_embedder_->GetMetadataExtractor()->GetModelMetadata()->UnPackTo(
      &model_metadata_t);
  // Sanity checks.
  if (model_metadata_t.subgraph_metadata.size() != 1) {
    return absl::InternalError(
        absl::StrFormat("Expected exactly one subgraph metadata, found %d.",
                        model_metadata_t.subgraph_metadata.size()));
  }
  if (model_metadata_t.subgraph_metadata[0]->output_tensor_metadata.size() !=
      1) {
    return absl::InternalError(absl::StrFormat(
        "Expected exactly one output tensor metadata, found %d.",
        model_metadata_t.subgraph_metadata[0]->output_tensor_metadata.size()));
  }

  // Set the relevant fields for the new model.
  model_metadata_t.name = name_;
  model_metadata_t.description = description_;
  model_metadata_t.author = author_;
  model_metadata_t.version = version_;
  model_metadata_t.license = license_;

  // Add the associated files.
  for (auto it = associated_files_.begin(); it != associated_files_.end();
       it++) {
    auto associated_file_t = std::make_unique<tflite::AssociatedFileT>();
    associated_file_t->name = it->first;
    associated_file_t->type = tflite::AssociatedFileType_DESCRIPTIONS;
    model_metadata_t.associated_files.push_back(std::move(associated_file_t));
  }

  // Build minimalistic output tensor metadata.
  auto tensor_metadata_t = std::make_unique<tflite::TensorMetadataT>();

  // Add the labelmap file.
  auto associated_file_t = std::make_unique<tflite::AssociatedFileT>();
  associated_file_t->name = kLabelMapFilename;
  associated_file_t->type = tflite::AssociatedFileType_TENSOR_AXIS_LABELS;
  tensor_metadata_t->associated_files.push_back(std::move(associated_file_t));

  // Replace output tensor metadata.
  model_metadata_t.subgraph_metadata[0]->output_tensor_metadata[0] =
      std::move(tensor_metadata_t);

  // Pack metadata.
  FlatBufferBuilder fbb;
  fbb.Finish(tflite::ModelMetadata::Pack(fbb, &model_metadata_t),
             tflite::ModelMetadataIdentifier());

  // Populate metadata.
  ASSIGN_OR_RETURN(
      auto metadata_populator,
      ModelMetadataPopulator::CreateFromModelBuffer(buffer_data, buffer_size));
  metadata_populator->LoadMetadata(
      reinterpret_cast<char*>(fbb.GetBufferPointer()), fbb.GetSize());

  // Add the labelmap file to the map of associated files and load them in the
  // metadata populator.
  std::string labelmap = absl::StrJoin(labels_, "\n");
  associated_files_.insert({{kLabelMapFilename, labelmap}});
  metadata_populator->LoadAssociatedFiles(associated_files_);
  return metadata_populator->Populate();
}

tflite::support::StatusOr<ExternalFile> ModelBuilder::BuildModel() {
  // Sanity checks.
  if (feature_vectors_.size() < 2) {
    return absl::FailedPreconditionError(
        absl::StrFormat("Expected at least two labeled images to have been "
                        "provided through `AddLabeledImage`, found %d.",
                        feature_vectors_.size()));
  }
  if (labels_.size() != feature_vectors_.size()) {
    return absl::InternalError(
        "Expected same number of labels and feature vectors.");
  }
  FlatBufferBuilder fbb;
  ASSIGN_OR_RETURN(labels_,
                   tflite_cbr_builder_->BuildCbRModel(
                       *model_->GetModel(), feature_vectors_, labels_, &fbb));

  // Populate metadata.
  ExternalFile model_external_file;
  ASSIGN_OR_RETURN(
      *model_external_file.mutable_file_content(),
      PopulateMetadata(reinterpret_cast<char*>(fbb.GetBufferPointer()),
                       fbb.GetSize()));

  // Reset state and return.
  name_ = "";
  description_ = "";
  author_ = "";
  version_ = "";
  license_ = "";
  associated_files_ = {};
  labels_.clear();
  feature_vectors_.clear();
  return model_external_file;
}

}  // namespace cbr
}  // namespace examples
}  // namespace tflite
