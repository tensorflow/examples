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

#import "ios/ImageClassifierBuilder/ModelTrainingUtils.h"

#import <CoreVideo/CoreVideo.h>
#import <UIKit/UIKit.h>

#include <cstdint>
#include "absl/status/status.h"
#include "flatbuffers/util.h"
#import "ios/ImageClassifierBuilder/NSData+PixelBuffer.h"
#import "ios/ImageClassifierBuilder/NSString+AbseilStringView.h"
#import "ios/ImageClassifierBuilder/UIImage+CoreVideo.h"
#include "lib/model_builder.h"
#include "tensorflow_lite_support/cc/port/statusor.h"
#include "tensorflow_lite_support/cc/task/core/proto/external_file_proto_inc.h"
#include "tensorflow_lite_support/cc/task/vision/core/frame_buffer.h"
#include "tensorflow_lite_support/cc/task/vision/proto/image_embedder_options_proto_inc.h"
#include "tensorflow_lite_support/cc/task/vision/utils/frame_buffer_common_utils.h"

using ::tflite::task::core::ExternalFile;
using ::tflite::task::vision::FrameBuffer;
using ::tflite::task::vision::CreateFromRgbaRawBuffer;
using ::tflite::task::vision::ImageEmbedderOptions;
using ::tflite::examples::cbr::ModelBuilder;

static NSString *const kEmbeddingModelName = @"imagenet-mobilenet_v3_small_100_224-feature_vector";
static NSString *const kEmbeddingModelExtension = @"tflite";

@implementation ModelTrainingUtils

+ (void)trainModelWithName:(NSString *)name
               description:(NSString *)description
                    author:(NSString *)author
                   version:(NSString *)version
                   license:(NSString *)license
                    labels:(NSArray<NSString *> *)labels
                imagePaths:(NSArray<NSString *> *)imagePaths
           outputModelPath:(NSString *)outputModelPath {
  NSParameterAssert(name.length > 0);
  NSParameterAssert(description.length > 0);
  NSParameterAssert(author.length > 0);
  NSParameterAssert(version.length > 0);
  NSParameterAssert(license.length > 0);
  NSParameterAssert(labels.count == imagePaths.count);
  NSParameterAssert(labels.count >= 2);

  // Create ModelBuilder.
  ImageEmbedderOptions image_embedder_options;
  NSString *embeddingModelPath =
      [NSBundle.mainBundle pathForResource:kEmbeddingModelName ofType:kEmbeddingModelExtension];
  NSAssert(embeddingModelPath, @"Unable to locate image embedder model file.");
  image_embedder_options.mutable_model_file_with_metadata()->set_file_name(
      embeddingModelPath.UTF8String);
  std::unique_ptr<ModelBuilder> model_builder =
      *ModelBuilder::CreateFromImageEmbedderOptions(image_embedder_options);

  // Set the metadata.
  model_builder->SetMetadata(name.UTF8String, description.UTF8String, author.UTF8String,
                             version.UTF8String, license.UTF8String,
                             {{"model_type.txt", "IMAGE_CLASSIFIER"}});

  // Loop on images.
  for (NSInteger i = 0; i < imagePaths.count; i++) {
    // Decode image and load into a FrameBuffer.
    UIImage *image = [[UIImage alloc] initWithContentsOfFile:imagePaths[i]];
    NSAssert(image, @"The image couldn't be loaded: %@", imagePaths[i]);
    CVPixelBufferRef pixelBuffer = [image cbr_asNewPixelBuffer];
    NSData *data = [NSData cbr_RGBA8888DataFromPixelBuffer:pixelBuffer];
    std::unique_ptr<FrameBuffer> frame_buffer =
        CreateFromRgbaRawBuffer(static_cast<const uint8 *>(data.bytes),
                                {static_cast<int>(CVPixelBufferGetWidth(pixelBuffer)),
                                 static_cast<int>(CVPixelBufferGetHeight(pixelBuffer))});

    // Add to model builder.
    auto status = model_builder->AddLabeledImage(labels[i].UTF8String, *frame_buffer);
    NSAssert(status.ok(), @"Could not add labeled image to the model builder: %@",
             [NSString cbr_stringWithStringView:status.message()]);
  }

  // Finalize model building and write to file.
  ExternalFile output_external_file = *model_builder->BuildModel();
  BOOL fileSaved =
      flatbuffers::SaveFile(outputModelPath.UTF8String, output_external_file.file_content(),
                            /*binary=*/true);
  NSAssert(fileSaved, @"Could not save the trained model file to %@.", outputModelPath);
}

@end
