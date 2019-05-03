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

#import "TfliteWrapper.h"
#import <UIKit/UIKit.h>

#import <vector>
#include <sys/time.h>
#include <fstream>
#include <iostream>
#include <queue>

#include "tensorflow/lite/model.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/string_util.h"

NSString* modelFileName;

#pragma mark - Initializer
@interface TfliteWrapper() {
  std::unique_ptr<tflite::FlatBufferModel> model;
  tflite::ops::builtin::BuiltinOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> interpreter;
}

@end

@implementation TfliteWrapper

-(instancetype)initWithModelFileName:(NSString *)fileName {
  self = [super init];
  if (self) {
    modelFileName = fileName;
  }

  return  self;
}

#pragma mark - Model and Interpreter Setup Methods
-(BOOL)setUpModelAndInterpreter {
  if (![self initializeFlatBufferModel]) {
    return NO;
  }

  if (![self initializeInterpreter]) {
    return NO;
  }

  return YES;
}

/**
 This method initializes the flat buffer model.
 */
-(BOOL)initializeFlatBufferModel {
  if (!modelFileName) {
    return NO;
  }

  // Obtains path to tflite model.
  NSString* graph_path = [TfliteWrapper filePathForResourceName:modelFileName extension:@"tflite"];
  if (!graph_path) {
    return NO;
  }

  // Tries to load tflite model.
  model = tflite::FlatBufferModel::BuildFromFile([graph_path UTF8String]);
  if (!model) {
    NSLog(@"Failed to mmap model %@", graph_path);
    return NO;
  }
  NSLog(@"Loaded model %@", graph_path);
  return YES;
}

/**
 This method initializes the interpreter and allocates the tensors.
 */
-(BOOL)initializeInterpreter {
  // Tries to construct interpreter.
  tflite::InterpreterBuilder(*model, resolver)(&interpreter);

  if (!interpreter) {
    NSLog(@"Failed to construct interpreter");
    return NO;
  }

  //Tries to allocate input and output tensors.
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    NSLog(@"Failed to construct interpreter");
    return NO;
  }

  return YES;
}

-(Float32 *)floatInputTensorAtIndex:(int) index {
  //Gets a reference to the tensor at a specified index
  int input = interpreter->inputs()[index];
  Float32* output = interpreter->typed_tensor<Float32>(input);
  return output;
}

-(int *)intInputTensorAtIndex:(int) index {
  // Gets a reference to the tensor at a specified index
  int input = interpreter->inputs()[index];
  int* output = interpreter->typed_tensor<int>(input);

  return output;
}

-(void)reshapeTensorAtIndex:(int)index toShape: (int[]) shape shapeCount: (int)count {
  std::vector<int> sizes;
  // Gets tensor number
  int input = interpreter->inputs()[index];

  // Resizes the tensor
  interpreter->ResizeInputTensor(input, sizes);
}


-(BOOL)invokeInterpreter {
  if (interpreter->Invoke() != kTfLiteOk) {
    NSLog(@"Failed to invoke!");
    return false;
  }

  return true;
}

-(float *)outputTensorDataWithIndex:(int) index {
  return interpreter->typed_output_tensor<float>(index);
}

#pragma mark - Thread Handling Methods
-(void)setNumberOfThreads:(int)threadCount {
  interpreter->SetNumThreads(threadCount);
}

+(NSString *)filePathForResourceName: (NSString*) name extension:(NSString*) extension {
  NSString* filePath = [[NSBundle mainBundle] pathForResource:name ofType:extension];
  if (filePath == NULL) {
    NSLog(@"Couldn't find \"%@.%@\" in bundle", name, extension);
  }
  return filePath;
}

@end
