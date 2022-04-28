// Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
#import "TFLAudioRecord.h"
#import "TFLRingBuffer.h"

NS_ASSUME_NONNULL_BEGIN

/** A wrapper class to store input audio used in on-device machine learning. */
NS_SWIFT_NAME(AudioTensor)
@interface TFLAudioTensor : NSObject

/** Audio format specifying the number of channels and sample rate supported. */
@property(nonatomic, readonly) TFLAudioFormat *audioFormat;

@property(nonatomic, readonly) TFLRingBuffer *ringBuffer;

/**
 * Initializes TFLAudioTensor with a TFLAudioFormat and sample countl.
 *
 * @discussion The created instance stores data in a ring buffer of size sampleCount *
 * TFLAudioFormat.channelCount.
 *
 * @param format An audio format of type TFLAudioFormat.
 * @seealso TFLAudioFormat
 *
 * @param sampleCount The number of samples this TFLAudioTensor instance can store at any given
 * time. The sampleCount provided will be used to calculate the buffer size needed by multiplying
 * with channelCount of format.
 *
 * @return An instance of TFLAudioFormat
 */
- (instancetype)initWithAudioFormat:(TFLAudioFormat *)format sampleCount:(NSUInteger)sampleCount;

/**
 * Convenience method to load the of  TFLAudioRecord into TFLTensorAudio.
 *
 * @seealso TFLAudioRecord
 *
 * @discussion You must make sure that the  audio formats of TFLAudioRecord and the current
 * TFLTensorAudio match.
 *
 * @param audioRecord  A TFLAudioRecord.
 *
 * @return A boolean indicating if the load operation succeded.
 */
- (BOOL)loadAudioRecord:(TFLAudioRecord *)audioRecord
              withError:(NSError **)error NS_SWIFT_NAME(loadAudioRecord(audioRecord:));

/**
 * This function loads the TFLAudioTensor ring buffer with a the provided buffer.
 *
 * @discussion New data from the input buffer is appended to the end of the buffer by shifting out
 * any old data from the beginning of the buffer if need be to make space. If the size of the new
 * data to be copied is more than the capacity of TFLAudioTensor's buffer, only the most recent data
 * of the TFLAudioTensor's buffer size will be copied from the input buffer .
 *
 * @seealso TFLAudioRecord
 *
 *
 * @param sourceBuffer  A buffer of type TFLFloatBuffer output by TFLAudioRecord. You must make sure
 * that the buffer size and audio format of TFLAudioRecord matches the the format. For multi-channel
 * input, the array is interleaved.
 * @param offset Starting position in the sorce buffer.
 * @param size The number of  values to be copied.
 *
 * @return An instance of TFLAudioFormat
 */
- (BOOL)loadBuffer:(TFLFloatBuffer *)sourceBuffer
            offset:(NSInteger)offset
              size:(NSInteger)size
             error:(NSError **)error;

@end

NS_ASSUME_NONNULL_END
