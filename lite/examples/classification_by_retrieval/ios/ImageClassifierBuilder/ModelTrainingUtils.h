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

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@class ModelMetadata;

@interface ModelTrainingUtils : NSObject

/// Trains a new classification model based on the labeled images.
/// @param name The name of the model. This must be not empty.
/// @param description The description of the model. This must be not empty.
/// @param author The author of the model. This must be not empty.
/// @param version The version of the model. This must be not empty.
/// @param license The license of the model. This must be not empty.
/// @param labels The list of labels. The count must match imagePaths'.
/// @param imagePaths The list of image paths. The count must match labels'.
/// @param outputModelPath The path on disk where to store the newly created model.
+ (void)trainModelWithName:(NSString *)name
               description:(NSString *)description
                    author:(NSString *)author
                   version:(NSString *)version
                   license:(NSString *)license
                    labels:(NSArray<NSString *> *)labels
                imagePaths:(NSArray<NSString *> *)imagePaths
           outputModelPath:(NSString *)outputModelPath
    NS_SWIFT_NAME(trainModel(name:description:author:version:license:labels:imagePaths:outputModelPath:));

@end

NS_ASSUME_NONNULL_END
