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

#import "TFLBaseOptions.h"
#import "TFLClassificationOptions.h"
#import "TFLClassificationResult.h"
#import "TFLAudioTensor.h"

NS_ASSUME_NONNULL_BEGIN

/**
 * Options to configure TFLImageClassifier.
 */
NS_SWIFT_NAME(AudioClassifierOptions)
@interface TFLAudioClassifierOptions : NSObject

/**
 * Base options that is used for creation of any type of task.
 * @seealso TFLBaseOptions
 */
@property(nonatomic, copy) TFLBaseOptions *baseOptions;

/**
 * Options that configure the display and filtering of results.
 * @seealso TFLClassificationOptions
 */
@property(nonatomic, copy) TFLClassificationOptions *classificationOptions;

/**
 * Initializes TFLImageClassifierOptions with the model path set to the specified path to a model
 * file.
 * @description The external model file, must be a single standalone TFLite file. It could be packed
 * with TFLite Model Metadata[1] and associated files if exist. Fail to provide the necessary
 * metadata and associated files might result in errors. Check the [documentation]
 * (https://www.tensorflow.org/lite/convert/metadata) for each task about the specific requirement.
 *
 * @param modelPath Path to a TFLite model file.
 *
 * @return An instance of TFLImageClassifierOptions set to the specified
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
 * Creates TFLAudioClassifier from a model file and specified options .
 *
 * @param options TFLAudioClassifierOptions instance with the necessary
 * properties set.
 *
 * @return A TFLAudioClassifier instance.
 */
- (nullable instancetype)initWithModelPath:(NSString *)modelPath error:(NSError **)error;
- (nullable instancetype)initWithOptions:(TFLAudioClassifierOptions *)options error:(NSError **)error;

- (nullable TFLAudioTensor *)createInputAudioTensorWithError:(NSError **)error;

- (nullable TFLAudioRecord *)createAudioRecordWithError:(NSError **)error;

- (NSInteger)requiredBufferSizeWithError:(NSError **)error;


+ (instancetype)new NS_UNAVAILABLE;
- (instancetype)init NS_UNAVAILABLE;

/**
 * Performs classification on a TensorAudio input, returns an array of
 * categorization results where each member in the array is an array of
 * TFLClass objects for each classification head.
 * This method currently supports inference on only following type of images:
 * 1. RGB and RGBA images for GMLImageSourceTypeImage.
 * 2. kCVPixelFormatType_32BGRA for GMLImageSourceTypePixelBuffer and
 *    GMLImageSourceTypeSampleBuffer. If you are using AVCaptureSession to setup
 *    camera and get the frames for inference, you must request for this format
 *    from AVCaptureVideoDataOutput. Otherwise your classification
 *    results will be wrong.
 *
 * @param image input to the model.
 *
 * @return An NSArray<NSArray<TFLClass *>*> * of classification results.
 */
- (nullable TFLClassificationResult *)classifyWithAudioTensor:(TFLAudioTensor *)audioTensor
                                                     error:(NSError *_Nullable *)error
    NS_SWIFT_NAME(classify(audioTensor:));


@end

NS_ASSUME_NONNULL_END
