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

#import <vector>
#include <sys/time.h>
#include <fstream>
#include <iostream>
#include <queue>

#include "tensorflow/contrib/lite/kernels/register.h"
#include "tensorflow/contrib/lite/model.h"

#define LOG(x) std::cerr

NSString* modelFileName;

@interface TfliteWrapper() {
  std::unique_ptr<tflite::FlatBufferModel> model;
  tflite::ops::builtin::BuiltinOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> interpreter;
}
@end

@implementation TfliteWrapper

#pragma mark - Initializer
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
-(BOOL) initializeFlatBufferModel {
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
    LOG(FATAL) << "Failed to mmap model " << graph_path;
    return NO;
  }
  LOG(INFO) << "Loaded model " << graph_path;
  return YES;
}

/**
 This method initializes the interpreter and allocates the tensors.
 */
-(BOOL) initializeInterpreter {
  //Tries to construct interpreter.
  tflite::InterpreterBuilder(*model, resolver)(&interpreter);

  if (!interpreter) {
    LOG(FATAL) << "Failed to construct interpreter";
    return NO;
  }

  //Tries to allocate input and output tensors.
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    LOG(FATAL) << "Failed to allocate tensors!";
    return NO;
  }

  return YES;
}

#pragma MARK - Inference Methods
-(Float32 *)inputTensorAtIndex:(int)index {
  //Gets a reference to the tensor at a specified index
  return interpreter->typed_input_tensor<float>(index);
}

-(float *)outputTensorAtIndex:(int)index {
  //Returns the output tensor.
  return interpreter->typed_output_tensor<float>(index);
}

-(BOOL)invokeInterpreter {
  //Invokes the interpreter
  if (interpreter->Invoke() != kTfLiteOk) {
    NSLog(@"Failed to invoke!");
    return false;
  }

  return true;
}


#pragma mark - Thread Handling Methods
-(void)setNumberOfThreads:(int)threadCount {
  interpreter->SetNumThreads(threadCount);
}

+(NSString *)filePathForResourceName: (NSString*) name extension:(NSString*) extension {
  NSString* filePath = [[NSBundle mainBundle] pathForResource:name ofType:extension];
  if (filePath == NULL) {
    LOG(FATAL) << "Couldn't find '" << [name UTF8String] << "." << [extension UTF8String]
    << "' in bundle.";
  }
  return filePath;
}

@end
