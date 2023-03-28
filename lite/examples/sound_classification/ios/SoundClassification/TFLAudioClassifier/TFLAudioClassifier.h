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
#import <Foundation/Foundation.h>

#import "TFLAudioTensor.h"
#import "TFLBaseOptions.h"
#import "TFLClassificationOptions.h"
#import "TFLClassificationResult.h"

NS_ASSUME_NONNULL_BEGIN

/**
 * Options to configure TFLAudioClassifier.
 */
NS_SWIFT_NAME(AudioClassifierOptions)
@interface TFLAudioClassifierOptions : NSObject

/**
 * Base options that is used for creation of any type of task.
 */
@property(nonatomic, copy) TFLBaseOptions *baseOptions;

/**
 * Options that configure the display and filtering of results.
 */
@property(nonatomic, copy) TFLClassificationOptions *classificationOptions;

/**
 * Initializes TFLAudioClassifierOptions with the model path set to the specified path to a model
 * file.
 * @discussion The external model file, must be a single standalone TFLite file. It could be packed
 * with TFLite Model Metadata[1] and associated files if exist. Fail to provide the necessary
 * metadata and associated files might result in errors. Check the [documentation]
 * (https://www.tensorflow.org/lite/convert/metadata) for each task about the specific requirement.
 *
 * @param modelPath Path to a TFLite model file.
 *
 * @return An instance of TFLAudioClassifierOptions set to the specified
 * modelPath.
 */
- (instancetype)initWithModelPath:(NSString *)modelPath;

- (instancetype)init NS_UNAVAILABLE;

+ (instancetype)new NS_UNAVAILABLE;

@end

/**
 * A TensorFlow Lite Task Audio Classifiier.
 */
NS_SWIFT_NAME(AudioClassifier)
@interface TFLAudioClassifier : NSObject

/**
 * Creates TFLAudioClassifier from a model file at location modelPath .
 *
 * @discussion The external model file, must be a single standalone TFLite file. It could be packed
 * with TFLite Model Metadata[1] and associated files if exist. Fail to provide the necessary
 * metadata and associated files might result in errors. Check the [documentation]
 * (https://www.tensorflow.org/lite/convert/metadata) for each task about the specific requirement.
 *
 * @param modelPath Path to the model file.
 *
 * @param error Address to an NSError * to populate the reason for failure to create
 * TFLAudioClassifier if the method returns nil.
 *
 * @return A TFLAudioClassifier instance if modelPath points to a model file that meets the
 * specifications. nil is returned if initialisation is unsuccessful.
 */
- (nullable instancetype)initWithModelPath:(NSString *)modelPath error:(NSError **)error;

/**
 * Creates TFLAudioClassifier from TFLAudioClassifierOptions.
 * @param options TFLAudioClassifierOptions specifying the modelPath to a model file and other
 * custom options for configuring the TFLAudioClassifier.
 *
 * @param error Address to an NSError * to populate the reason for failure to create
 * TFLAudioClassifier if the method returns nil.
 *
 * @return A TFLAudioClassifier instance if initialization is successful or nil in case of failure.
 */
- (nullable instancetype)initWithOptions:(TFLAudioClassifierOptions *)options
                                   error:(NSError *_Nullable *)error;

/**
 * Creates a TFLAudioTensor instance to store input audio samples.
 *
 * @param error Address to an NSError * to populate the reason for failure to create TFLAudioTensor
 * if the method returns nil.
 *
 * @return A TFLAudioTensor with the same buffer size as the model input tensor if creation is
 * successful otherwise nil.
 */
- (nullable TFLAudioTensor *)createInputAudioTensorWithError:(NSError **)error;

/**
 * Creates an TFLAudioRecord instance to tap audio data from input audio stream. The returned
 * TFLAudioRecordinstance instance is initialized with the same audio format and 2 * input buffer
 * size required by the model.
 *
 * @discussion The  client needs to call TFLAudioRecord's - (BOOL)startRecordingWithError:(NSError
 * **)error to start tapping the input audio stream.
 *
 * @param error Address to an NSError * to populate the reason for failure to create TFLAudioRecord
 * if the method returns nil.
 *
 * @return A TFLAudioRecord instance if creation is successful or nil in case of failure.
 */
- (nullable TFLAudioRecord *)createAudioRecordWithError:(NSError **)error;

/**
 * Returns the required input buffer size in number elements or zero in case of failure to determine
 * the input buffer size.
 *
 * @param error Address to an NSError * to populate the reason for failure to determine the input
 * buffer size.
 *
 * @return Required input buffer size in number elements or zero in case of failure to determine the
 * input buffer size
 */
- (NSInteger)requiredBufferSizeWithError:(NSError **)error;

/**
 * Returns the  TFLAudioFormat matching the model requirements
 *
 * @param error Address to an NSError * to populate the reason for failure to return a
 * TFLAudioFormat if the method returns nil.
 *
 * @return Returns the TFLAudioFormat matching the model requirements or nil in case of failure to
 * determine the required TFLAudioFormat.
 */
- (TFLAudioFormat *)requiredTensorFormatWithError:(NSError **)error;

/**
 * Performs classification on an array of audio samples held by TFLAudioTensor input and returns a
 * TFLClassificationResult which holds results of classification for each head of the model.
 *
 * @param audioTensor TFLAudioTensor to be classified.
 *
 * @return TFLClassificationResult which holds results of classification for each head of the model.
 */
- (nullable TFLClassificationResult *)classifyWithAudioTensor:(TFLAudioTensor *)audioTensor
                                                        error:(NSError **)error
    NS_SWIFT_NAME(classify(audioTensor:));

+ (instancetype)new NS_UNAVAILABLE;
- (instancetype)init NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END
