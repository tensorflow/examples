/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 ==============================================================================*/
#import "TFLAudioClassifier.h"
#import "TFLAudioTensor+Utils.h"
#import "TFLBaseOptions+Helpers.h"
#import "TFLClassificationOptions+Helpers.h"
#import "TFLClassificationResult+Helpers.h"
#import "TFLCommon.h"
#import "TFLCommonUtils.h"

@import TensorFlowLiteTaskAudioC;

@interface TFLAudioClassifier ()
/** Audio Classifier backed by C API */
@property(nonatomic) TfLiteAudioClassifier *audioClassifier;
@end

@implementation TFLAudioClassifierOptions
@synthesize baseOptions;
@synthesize classificationOptions;

- (instancetype)init {
  self = [super init];
  if (self) {
    self.baseOptions = [[TFLBaseOptions alloc] init];
    self.classificationOptions = [[TFLClassificationOptions alloc] init];
  }
  return self;
}

- (instancetype)initWithModelPath:(NSString *)modelPath {
  self = [self init];
  if (self) {
    self.baseOptions.modelFile.filePath = modelPath;
  }
  return self;
}

@end

@implementation TFLAudioClassifier
- (void)dealloc {
  TfLiteAudioClassifierDelete(_audioClassifier);
}

- (instancetype)initWithAudioClassifier:(TfLiteAudioClassifier *)audioClassifier {
  self = [super init];
  if (self) {
    _audioClassifier = audioClassifier;
  }
  return self;
}

- (nullable instancetype)initWithCOptions:(TfLiteAudioClassifierOptions)cOptions
                                    error:(NSError **)error {
  TfLiteSupportError *cCreateClassifierError = NULL;
  TfLiteAudioClassifier *cAudioClassifier =
      TfLiteAudioClassifierFromOptions(&cOptions, &cCreateClassifierError);

  // Populate iOS error if TfliteSupportError is not null and afterwards delete it.
  if (![TFLCommonUtils checkCError:cCreateClassifierError toError:error]) {
    TfLiteSupportErrorDelete(cCreateClassifierError);
  }

  // Return nil if classifier evaluates to nil. If an error was generted by the C layer, it has
  // already been populated to an NSError and deleted before returning from the method.
  if (!cAudioClassifier) {
    return nil;
  }

  return [self initWithAudioClassifier:cAudioClassifier];
}

- (nullable instancetype)initWithOptions:(TFLAudioClassifierOptions *)options
                                   error:(NSError **)error {
  if (!options) {
    [TFLCommonUtils createCustomError:error
                             withCode:TFLSupportErrorCodeInvalidArgumentError
                          description:@"modelPath argument cannot be nil."];
    return nil;
  }

  TfLiteAudioClassifierOptions cOptions = TfLiteAudioClassifierOptionsCreate();

  if (![options.classificationOptions copyToCOptions:&(cOptions.classification_options)
                                               error:error]) {
    [options.classificationOptions
        deleteAllocatedMemoryOfClassificationOptions:&(cOptions.classification_options)];
    return nil;
  }

  [options.baseOptions copyToCOptions:&(cOptions.base_options)];

  TFLAudioClassifier *audioClassifier = [self initWithCOptions:cOptions error:error];

  [options.classificationOptions
      deleteAllocatedMemoryOfClassificationOptions:&(cOptions.classification_options)];

  return audioClassifier;
}

- (nullable instancetype)initWithModelPath:(NSString *)modelPath error:(NSError **)error {
  if (!modelPath) {
    [TFLCommonUtils createCustomError:error
                             withCode:TFLSupportErrorCodeInvalidArgumentError
                          description:@"modelPath argument cannot be nil."];
    return nil;
  }

  TfLiteAudioClassifierOptions cOptions = TfLiteAudioClassifierOptionsCreate();
  [TFLBaseOptions copyModelPath:modelPath toCOptions:&(cOptions.base_options)];

  return [self initWithCOptions:cOptions error:error];
}

- (nullable TFLClassificationResult *)classifyWithAudioTensor:(TFLAudioTensor *)audioTensor
                                                        error:(NSError **)error {
  if (!audioTensor) {
    [TFLCommonUtils createCustomError:error
                             withCode:TFLSupportErrorCodeInvalidArgumentError
                          description:@"audioTensor argument cannot be nil."];
    return nil;
  }

  TfLiteAudioBuffer cAudioBuffer =
      [audioTensor cAudioBufferFromFloatBuffer:[audioTensor.ringBuffer floatBuffer]];

  TfLiteSupportError *classifyError = NULL;
  TfLiteClassificationResult *cClassificationResult =
      TfLiteAudioClassifierClassify(_audioClassifier, &cAudioBuffer, &classifyError);

  // Populate iOS error if C Error is not null and afterwards delete it.
  if (![TFLCommonUtils checkCError:classifyError toError:error]) {
    TfLiteSupportErrorDelete(classifyError);
  }

  // Return nil if C result evaluates to nil. If an error was generted by the C layer, it has
  // already been populated to an NSError and deleted before returning from the method.
  if (!cClassificationResult) {
    return nil;
  }

  TFLClassificationResult *classificationResult =
      [TFLClassificationResult classificationResultWithCResult:cClassificationResult];
  TfLiteClassificationResultDelete(cClassificationResult);

  return classificationResult;
}

- (TFLAudioFormat *)requiredTensorFormatWithError:(NSError **)error {
  TfLiteSupportError *getAudioFormatError = nil;
  TfLiteAudioFormat *cFormat =
      TfLiteAudioClassifierGetRequiredAudioFormat(_audioClassifier, &getAudioFormatError);

  if (![TFLCommonUtils checkCError:getAudioFormatError toError:error]) {
    TfLiteSupportErrorDelete(getAudioFormatError);
  }

  if (!cFormat) {
    return nil;
  }

  TFLAudioFormat *format = [[TFLAudioFormat alloc] initWithChannelCount:cFormat->channels
                                                             sampleRate:cFormat->sample_rate];

  TfLiteAudioFormatDelete(cFormat);

  return format;
}

- (NSInteger)requiredBufferSizeWithError:(NSError **)error {
  NSInteger bufferSize = TfLiteAudioClassifierGetRequiredInputBufferSize(_audioClassifier, NULL);
  if (bufferSize <= 0) {
    [TFLCommonUtils
        createCustomError:error
                 withCode:TFLSupportErrorCodeUnspecifiedError
              description:@"Some error occured while trying to create input audio tensor."];
  }
  return bufferSize;
}

- (TFLAudioTensor *)createInputAudioTensorWithError:(NSError **)error {
  TFLAudioFormat *format = [self requiredTensorFormatWithError:error];

  if (!format) {
    return nil;
  }

  NSInteger bufferSize = [self requiredBufferSizeWithError:error];

  if (bufferSize <= 0) {
    return nil;
  }

  return [[TFLAudioTensor alloc] initWithAudioFormat:format
                                         sampleCount:bufferSize / format.channelCount];
}

- (TFLAudioRecord *)createAudioRecordWithError:(NSError **)error {
  TFLAudioFormat *format = [self requiredTensorFormatWithError:error];

  if (!format) {
    return nil;
  }

  NSInteger bufferSize = [self requiredBufferSizeWithError:error];

  if (bufferSize <= 0) {
    return nil;
  }

  // The sample count of audio record should be strictly longer than audio tensor's so that
  // clients could run TFLAudioRecord's `startRecordingWithError:`
  // together with TFLAudioClassifier's `classifyWithAudioTensor:error:`
  return [[TFLAudioRecord alloc] initWithAudioFormat:format
                                         sampleCount:(bufferSize / format.channelCount) * 2
                                               error:error];
}

@end
