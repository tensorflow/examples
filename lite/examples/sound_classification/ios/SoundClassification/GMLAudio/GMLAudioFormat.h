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

NS_ASSUME_NONNULL_BEGIN

/**
 * Wraps a few constants describing the format of the incoming audio samples, namely number of
 * channels and the sample rate.
 */
@interface GMLAudioFormat : NSObject

@property (nonatomic, readonly) NSUInteger channelCount;
@property (nonatomic, readonly) NSUInteger sampleRate;

-(instancetype)initWithChannelCount:(NSUInteger)channelCount sampleRate:(NSUInteger)sampleRate;

/**
 * Initializes GMLAudioFormat with a default channel count oof 1 and the sample rate specified in the argument.
 *
 * @param sampleRate Sample rate.
 *
 * @return An instance of GMLAudioFormat
 */
-(instancetype)initWithSampleRate:(NSUInteger)sampleRate;

@end

NS_ASSUME_NONNULL_END
