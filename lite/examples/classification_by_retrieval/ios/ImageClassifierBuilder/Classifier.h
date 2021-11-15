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

#import <CoreVideo/CoreVideo.h>
#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@class Label;

/// A classifier infers labels from a given image. This wraps the C++ Image Classifier TFLite Task
/// API.
@interface Classifier : NSObject

/// Initializes a classifier, loading the given model.
/// @param URL The URL of the model file on disk.
- (instancetype)initWithModelURL:(NSURL *)modelURL NS_SWIFT_NAME(init(modelURL:));

/// MARK: Inference

/// Infers labels from the given image.
/// @param pixelBuffer The image, as a Core Video Pixel Buffer.
/// @return The list of labels.
- (NSArray<Label *> *)classifyPixelBuffer:(CVPixelBufferRef)pixelBuffer
    NS_SWIFT_NAME(classify(pixelBuffer:));

/// MARK: Model Infos

/// The name of the model.
@property(nonatomic, readonly) NSString *name;

/// The description of the model. (Note: `description` is already a method on NSObject, thus the
/// prefixing).
@property(nonatomic, readonly) NSString *modelDescription;

/// The author of the model.
@property(nonatomic, readonly) NSString *author;

/// The version of the model.
@property(nonatomic, readonly) NSString *version;

/// The license under which the model is released.
@property(nonatomic, readonly) NSString *license;

/// The labels in the labelmap. Note: this is not declared as a property because the implementation
/// is not trivial. It requires reading in the model file's metadata.
- (NSArray<NSString *> *)labels;

@end

/// Corresponds to a label from a classification result.
NS_SWIFT_NAME(Classifier.Label)
@interface Label : NSObject

/// The name of the label.
@property(nonatomic, readonly) NSString *name;

/// The score associated to the label. It can be any value, but is usually within [0, 1].
@property(nonatomic, readonly) float score;

/// Initializes a label.
/// @param name The name of the label.
/// @param score The score of the label.
- (instancetype)initWithName:(NSString *)name score:(float)score;

@end

NS_ASSUME_NONNULL_END
