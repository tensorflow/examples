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

/** Size of the buffer held by TFLAudioRecord. It ensures delivery of audio data of length bufferSize
 * arrays when you tap the input microphone. */
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
 * Taps the input of the on-device microphone and delivers the incoming audio data continuously in a
 * completion handler. Note that the completion handler delivers results on a background thread.
 *
 * @param completionHandler Completion handler delivers the status of audio record permission request, once it completes.
 *  If permission is not granted a n error is passed as the completion handler argument.
 *
 */
- (void)startRecordingWithCompletionHandler:
(void (^)(NSError *_Nullable error))completionHandler
NS_SWIFT_NAME(startRecording(_:));

- (void)stop;


/**
 * Returns the size number of elements in the TFLAudioRecord's buffer starting at offset.
 *
 * @param offset Offset inTFLAudioRecord's buffer from which elements are to be returned.
 *
 * @param size Number of elements to be returned.
 *
 * @returns A TFLFloatBuffer if offset + size is within the bounds of the TFLAudioRecord's buffer , otherwise nil.
 */
- (nullable TFLFloatBuffer *)readAtOffset:(NSUInteger)offset
                                 withSize:(NSUInteger)size
                                    error:(NSError *_Nullable *)error;

@end

NS_ASSUME_NONNULL_END
