// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#import <CoreImage/CoreImage.h>
#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>

/** This class provides Objective C wrapper methods on top of the C++ TensorflowLite library. This
 * wrapper is required because currently there is no interoperability between Swift and C++. This
 * wrapper is exposed to Swift via bridging so that the Tensorflow Lite methods can be called from
 * Swift.
 */
@interface TfliteWrapper : NSObject

/**
 This method initializes the TfliteWrapper with the specified model file.
 */
- (instancetype)initWithModelFileName:(NSString *)fileName;

/**
 This method initializes the interpreter of TensorflowLite library with the specified model file
 that performs the inference.
 */
- (BOOL)setUpModelAndInterpreter;

/**
 This method gets a reference to the input tensor at an index.
 */
- (uint8_t *)inputTensortAtIndex:(int)index;

/**
 This method performs the inference by invoking the interpreter.
 */
- (BOOL)invokeInterpreter;

/**
 This method gets the output tensor at a specified index.
 */
- (float *)outputTensorAtIndex:(int)index;

/**
 This method sets the number of threads used by the interpreter to perform inference.
 */
- (void)setNumberOfThreads:(int)threadCount;

@end
