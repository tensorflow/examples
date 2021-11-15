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

#import "ios/ImageClassifierBuilder/Classifier.h"

#include <cstdint>
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "flatbuffers/flatbuffers.h"
#import "ios/ImageClassifierBuilder/NSData+PixelBuffer.h"
#import "ios/ImageClassifierBuilder/NSString+AbseilStringView.h"
#include "tensorflow_lite_support/cc/task/core/base_task_api.h"
#include "tensorflow_lite_support/cc/task/vision/core/frame_buffer.h"
#include "tensorflow_lite_support/cc/task/vision/image_classifier.h"
#include "tensorflow_lite_support/cc/task/vision/proto/classifications_proto_inc.h"
#include "tensorflow_lite_support/cc/task/vision/proto/image_classifier_options_proto_inc.h"
#include "tensorflow_lite_support/cc/task/vision/utils/frame_buffer_common_utils.h"
#include "tensorflow_lite_support/metadata/cc/metadata_extractor.h"
#include "tensorflow_lite_support/metadata/metadata_schema_generated.h"

using ::tflite::task::vision::ClassificationResult;
using ::tflite::task::vision::Classifications;
using ::tflite::task::vision::CreateFromRgbaRawBuffer;
using ::tflite::task::vision::FrameBuffer;
using ::tflite::task::vision::ImageClassifier;
using ::tflite::task::vision::ImageClassifierOptions;

/// Merges similar labels. It keeps the highest score per label.
/// Courtesy of mbrenon.
/// @param input The classifications from a classification result.
static Classifications PostProcessClassifications(const Classifications &input) {
  Classifications output;
  absl::flat_hash_set<std::string> labels;
  for (const auto &class_ : input.classes()) {
    const std::string &label = class_.class_name();
    // Input classes are sorted by decreasing score, so keep only the first occurrence for each
    // label, which will also be the one with highest score.
    if (!labels.contains(label)) {
      labels.emplace(label);
      *output.add_classes() = class_;
    }
  }
  return output;
}

/// Converts C++ results to a list of Objective-C Label objects.
/// @param classificationResult The C++ classification result.
static NSArray<Label *> *LabelsFromClassificationResult(
    const ClassificationResult &classificationResult) {
  NSMutableArray<Label *> *labels = [NSMutableArray array];
  for (const auto &classifications : classificationResult.classifications()) {
    const auto processedClassifications = PostProcessClassifications(classifications);
    for (const auto &label : processedClassifications.classes()) {
      NSString *name = [NSString stringWithUTF8String:label.class_name().c_str()];
      [labels addObject:[[Label alloc] initWithName:name score:label.score()]];
    }
  }
  return labels;
}

@implementation Classifier {
  /// The underlying C++ image classifier.
  std::unique_ptr<ImageClassifier> _image_classifier;
}

- (instancetype)initWithModelURL:(NSURL *)modelURL {
  self = [super init];
  if (self) {
    ImageClassifierOptions options;
    options.mutable_model_file_with_metadata()->set_file_name(modelURL.path.UTF8String);
    auto imageClassifier = ImageClassifier::CreateFromOptions(options);
    NSAssert(imageClassifier.ok(), @"Couldn't create classifier: %@",
             [NSString cbr_stringWithStringView:imageClassifier.status().message()]);
    _image_classifier = std::move(imageClassifier.value());
  }
  return self;
}

/// MARK: Inference

- (NSArray<Label *> *)classifyPixelBuffer:(CVPixelBufferRef)pixelBuffer {
  NSData *data = [NSData cbr_RGBA8888DataFromPixelBuffer:pixelBuffer];
  const uint8 *input = static_cast<const uint8 *>(data.bytes);
  int width = static_cast<int>(CVPixelBufferGetWidth(pixelBuffer));
  int height = static_cast<int>(CVPixelBufferGetHeight(pixelBuffer));
  std::unique_ptr<FrameBuffer> frame_buffer = CreateFromRgbaRawBuffer(input, {width, height});
  auto result = *_image_classifier->Classify(*frame_buffer);
  return LabelsFromClassificationResult(result);
}

/// MARK: Model Infos

- (NSString *)name {
  auto metadata = _image_classifier->GetMetadataExtractor()->GetModelMetadata();
  return [NSString stringWithUTF8String:metadata->name()->c_str()];
}

- (NSString *)modelDescription {
  auto metadata = _image_classifier->GetMetadataExtractor()->GetModelMetadata();
  return [NSString stringWithUTF8String:metadata->description()->c_str()];
}

- (NSString *)author {
  auto metadata = _image_classifier->GetMetadataExtractor()->GetModelMetadata();
  return [NSString stringWithUTF8String:metadata->author()->c_str()];
}

- (NSString *)version {
  auto metadata = _image_classifier->GetMetadataExtractor()->GetModelMetadata();
  return [NSString stringWithUTF8String:metadata->version()->c_str()];
}

- (NSString *)license {
  auto metadata = _image_classifier->GetMetadataExtractor()->GetModelMetadata();
  return [NSString stringWithUTF8String:metadata->license()->c_str()];
}

- (NSArray<NSString *> *)labels {
  auto labelmap = _image_classifier->GetMetadataExtractor()->GetAssociatedFile("labelmap.txt");
  NSAssert(labelmap.ok(), @"Couldn’t extract the model's labelmap: %@",
           [NSString cbr_stringWithStringView:labelmap.status().message()]);
  std::vector<absl::string_view> raw_labels = absl::StrSplit(*labelmap, '\n');
  absl::flat_hash_set<std::string> unique_labels(raw_labels.begin(), raw_labels.end());
  NSMutableArray<NSString *> *labels = [NSMutableArray array];
  for (const auto& label : unique_labels) {
    [labels addObject:[NSString stringWithUTF8String:label.c_str()]];
  }
  return labels;
}

@end

@implementation Label

- (instancetype)initWithName:(NSString *)name score:(float)score {
  self = [super init];
  if (self) {
    _name = [name copy];
    _score = score;
  }
  return self;
}

/// Conveniently describes the label.
- (NSString *)description {
  return [NSString stringWithFormat:@"%@ (%.2f)", self.name, self.score];
}

@end
