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
#import "TFLAudioFormat.h"
#import "TFLFloatBuffer.h"

@import AVFoundation;

NS_ASSUME_NONNULL_BEGIN

/** A wrapper class to tap the device's microphone continuously. Currently this class only supports
 tapping the input node of the AVAudioEngine which emits audio data having only one channel.*/
NS_SWIFT_NAME(AudioRecord)
@interface TFLAudioRecord : NSObject

/** Audio format specifying the number of channels and sample rate supported. */
@property(nonatomic, readonly) TFLAudioFormat *audioFormat;

/** Size of the buffer held by TFLAudioRecord. It ensures delivery of audio data of length
 * bufferSize arrays when you tap the input microphone. */
@property(nonatomic, readonly) NSUInteger bufferSize;

/**
 * Initializes TFLAudioRecord with a TFLAudioFormat and sample count.
 *
 * @param format An audio format of type TFLAudioFormat.
 * @seealso TFLAudioFormat
 *
 * @param sampleCount The number of samples this TFLAudioRecord instance should delliver
 * continuously when you tap the on-device microphone. The tap callback will deliver arrays of size
 * sampleCount * bufferSize when you tap the microphone using
 * (checkAndStartTappingMicrophoneWithCompletionHandler:).
 *
 * @return An instance of TFLAudioRecord
 */
- (nullable instancetype)initWithAudioFormat:(TFLAudioFormat *)format
                                 sampleCount:(NSUInteger)sampleCount
                                       error:(NSError *_Nullable *)error;

/**
 * This function starts tapping the input audio samplles from the mic if audio record permissions
 * have been granted by the user. Before calling this function, you must use [AVAudioSession
 * sharedInstance]'s  - (void)requestRecordPermission:(void (^)(BOOL granted))response to acquire
 * record permissions. If  the user has denied permission or the permissions are undetermined, an
 * appropriate error is populated in the error pointer. The return value will be false in such cases
 * The internal buffer of TFLAudioRecord is of size bufferSize. bufferSize =
 * audioFormat.channelCount * sampleCount passed while initializing TFLAudioRecord. This buffer will
 * always have the most recent data samples acquired from the mic.  You can use:
 * - (nullable TFLFloatBuffer *)readAtOffset:(NSUInteger)offset withSize:(NSUInteger)size
 * error:(NSError *_Nullable *)error for getting the data from the buffer if audio recording has
 * started successfully.
 *
 * You can use - (void)stop to stop tapping the  mic input.
 *
 * @return Boolean value indicating if audio recording started successfully. If False and an address
 * to an error is passed in, the error will hold the reason for failure once the function returns.
 */
- (BOOL)startRecordingWithError:(NSError **)error NS_SWIFT_NAME(startRecording());

/**
 * Stops tapping the audio samples from input mic.
 */
- (void)stop;

/**
 * Returns the size number of elements in the TFLAudioRecord's buffer starting at offset.
 *
 * @param offset Offset inTFLAudioRecord's buffer from which elements are to be returned.
 *
 * @param size Number of elements to be returned.
 *
 * @returns A TFLFloatBuffer if offset + size is within the bounds of the TFLAudioRecord's buffer ,
 * otherwise nil.
 */
- (nullable TFLFloatBuffer *)readAtOffset:(NSUInteger)offset
                                 withSize:(NSUInteger)size
                                    error:(NSError *_Nullable *)error;

@end

NS_ASSUME_NONNULL_END
