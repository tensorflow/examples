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
#import "GMLAudioFormat.h"
#import "GMLRingBuffer.h"

@import AVFoundation;

NS_ASSUME_NONNULL_BEGIN

/** A wrapper class to tap the device's microphone continuously. Currently this class only supports
 tapping the input node of the AVAudioEngine which emits audio data having only one channel.*/
NS_SWIFT_NAME(AudioRecord)
@interface TFLAudioRecord : NSObject

/** Audio format specifying the number of channels and sample rate supported. */
@property(nonatomic, readonly) GMLAudioFormat *audioFormat;

/** Size of the buffer held by TFLAudioRecod. It ensures delivery of audio data of length bufferSize
 * arrays when you tap the input microphone. */
@property(nonatomic, readonly) NSUInteger bufferSize;

/**
 * Initializes TFLAudioRecord with a GMLAudioFormat and sample count.
 *
 * @param format An audio format of type GMLAudioFormat.
 * @seealso GMLAudioFormat
 *
 * @param sampleCount The number of samples this TFLAudioRecord instance should delliver
 * continuously when you tap the on-device microphone. The tap callback will deliver arrays of size
 * sampleCount * bufferSize when you tap the microphone using
 * (checkAndStartTappingMicrophoneWithCompletionHandler:).
 *
 * @return An instance of TFLAudioRecord
 */
- (nullable instancetype)initWithAudioFormat:(GMLAudioFormat *)format
                                 sampleCount:(NSUInteger)sampleCount
                                       error:(NSError *_Nullable *)error;

/**
 * Taps the input of the on-device microphone and delivers the incoming audio data continuously in a
 * completion handler. Note that the completion handler delivers results on a background thread.
 *
 * @param completionHandler Completion handler deliivers either a buffer of size bufferSize or an
 * error failing to do so .
 *
 */
- (void)checkPermissionsAndStartTappingMicrophoneWithCompletionHandler:
    (void (^)(GMLFloatBuffer *_Nullable buffer, NSError *_Nullable error))completionHandler;

- (void)stopTappingMicrophone;

@end

NS_ASSUME_NONNULL_END
