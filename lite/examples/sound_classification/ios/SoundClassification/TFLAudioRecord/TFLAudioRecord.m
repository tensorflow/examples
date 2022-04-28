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

#import "TFLAudioRecord.h"
#import "TFLAudioError.h"
#import "TFLRingBuffer.h"
#import "TFLUtils.h"

#define SUPPORTED_CHANNEL_COUNT 2

@implementation TFLAudioRecord {
  AVAudioEngine *_audioEngine;

  /* Specifying a custom buffer size on AVAUdioEngine while tapping does not take effect. Hence we
   * are storing the returned samples in a ring buffer to acheive the desired buffer size. If
   * specified buffer size is shorter than the buffer size supported by AVAUdioEngine only the most
   * recent data of the buffer of size bufferSize will be stored by the ring buffer. */
  TFLRingBuffer *_ringBuffer;
  dispatch_queue_t _conversionQueue;
  BOOL _tapStateSuccess;
  NSError *globalError;
}

- (nullable instancetype)initWithAudioFormat:(TFLAudioFormat *)audioFormat
                                 sampleCount:(NSUInteger)sampleCount
                                       error:(NSError *_Nullable *)error {
  self = [self init];
  if (self) {
    if (audioFormat.channelCount > SUPPORTED_CHANNEL_COUNT) {
      [TFLUtils
          createCustomError:error
                   withCode:TFLAudioErrorCodeWaitingForNewInputError
                description:@"The channel count provided does not match the supported "
                            @"channel count. Only upto 2 audio channels are currently supported."];
      return nil;
    }

    NSError *waitError = nil;
    [TFLUtils createCustomError:&waitError
                       withCode:TFLAudioErrorCodeWaitingForNewInputError
                    description:@"TFLAudioRecord hasn't started receiving samples from the audio "
                                @"input source. Please wait for the input."];

    globalError = waitError;
    _audioFormat = audioFormat;
    _audioEngine = [[AVAudioEngine alloc] init];
    _bufferSize = sampleCount * audioFormat.channelCount;
    _ringBuffer = [[TFLRingBuffer alloc] initWithBufferSize:sampleCount * audioFormat.channelCount];
    _conversionQueue =
        dispatch_queue_create("com.tflAudio.AudioConversionQueue", NULL);  // Serial Queue
  }
  return self;
}

/**
 * Uses AVAudioConverter to acheive the desired sample rate since the tap on the input node raises
 * an exception if any format other than the input nodes default format is specified. completion
 * handler structure is used as opposed to polling since this is the pattern followed by all iOS
 * APIs delivering continuous data on a background thread. Even if we implement a read method, it
 * will have to return data in a callback since installTapOnBus delivers data in a callback on a
 * different thread. There will also be extra overhead to ensure thread safety to make sure that
 * reads and writes happen on the same thread sine TFLAudioTensor buffer is meant to be non local.
 */
- (void)startTappingMicrophoneWithError:(NSError **)error {
  AVAudioNode *inputNode = [_audioEngine inputNode];
  AVAudioFormat *format = [inputNode outputFormatForBus:0];

  AVAudioFormat *recordingFormat =
      [[AVAudioFormat alloc] initWithCommonFormat:AVAudioPCMFormatFloat32
                                       sampleRate:self.audioFormat.sampleRate
                                         channels:(AVAudioChannelCount)self.audioFormat.channelCount
                                      interleaved:YES];

  AVAudioConverter *audioConverter = [[AVAudioConverter alloc] initFromFormat:format
                                                                     toFormat:recordingFormat];

  // Setting buffer size takes no effect on the input node. This class uses a ring buffer internally
  // to ensure the requested buffer size.
  [inputNode
      installTapOnBus:0
           bufferSize:(AVAudioFrameCount)self.bufferSize
               format:format
                block:^(AVAudioPCMBuffer *buffer, AVAudioTime *when) {
                  dispatch_async(self->_conversionQueue, ^{
                    // Capacity of converted PCM buffer is calculated in order to maintain the same
                    // latency as the input pcmBuffer.
                    AVAudioFrameCount capacity =
                        ceil(buffer.frameLength * recordingFormat.sampleRate / format.sampleRate);

                    AVAudioPCMBuffer *pcmBuffer = [[AVAudioPCMBuffer alloc]
                        initWithPCMFormat:recordingFormat
                            frameCapacity:capacity *
                                          (AVAudioFrameCount)self.audioFormat.channelCount];

                    AVAudioConverterInputBlock inputBlock =
                        ^AVAudioBuffer *_Nullable(AVAudioPacketCount inNumberOfPackets,
                                                  AVAudioConverterInputStatus *_Nonnull outStatus) {
                      *outStatus = AVAudioConverterInputStatus_HaveData;
                      return buffer;
                    };

                    NSError *conversionError = nil;
                    AVAudioConverterOutputStatus converterStatus =
                        [audioConverter convertToBuffer:pcmBuffer
                                                  error:&conversionError
                                     withInputFromBlock:inputBlock];

                    switch (converterStatus) {
                      case AVAudioConverterOutputStatus_HaveData: {
                        TFLFloatBuffer *floatBuffer =
                            [[TFLFloatBuffer alloc] initWithData:pcmBuffer.floatChannelData[0]
                                                            size:pcmBuffer.frameLength];

                        NSError *frameBufferProcessingError = nil;

                        if (pcmBuffer.frameLength == 0) {
                          [TFLUtils createCustomError:&frameBufferProcessingError
                                             withCode:TFLAudioErrorCodeInvalidArgumentError
                                          description:@"You may have to try with a different "
                                                      @"channel count or sample rate"];
                          self->globalError = frameBufferProcessingError;
                        } else if ((pcmBuffer.frameLength % recordingFormat.channelCount) != 0) {
                          [TFLUtils
                              createCustomError:&frameBufferProcessingError
                                       withCode:TFLAudioErrorCodeInvalidArgumentError
                                    description:
                                        @"You have passed an unsupported number of channels."];
                          self->globalError = frameBufferProcessingError;
                        } else if (![self->_ringBuffer loadBuffer:floatBuffer
                                                           offset:0
                                                             size:floatBuffer.size
                                                            error:&frameBufferProcessingError]) {
                          self->globalError = frameBufferProcessingError;
                        } else {
                          self->globalError = nil;
                        }
                        break;
                      }
                      case AVAudioConverterOutputStatus_Error:  // fall through
                      default: {
                        if (!conversionError) {
                          [TFLUtils
                              createCustomError:&conversionError
                                       withCode:TFLAudioErrorCodeAudioProcessingError
                                    description:@"Some error occurred during audio processing"];
                        }
                        self->globalError = conversionError;
                        break;
                      }
                    }
                  });
                }];

  [_audioEngine prepare];
  [_audioEngine startAndReturnError:error];
}

- (BOOL)startRecordingWithError:(NSError **)error {
  switch ([AVAudioSession sharedInstance].recordPermission) {
    case AVAudioSessionRecordPermissionDenied: {
      [TFLUtils createCustomError:error
                         withCode:TFLAudioErrorCodeRecordPermissionDeniedError
                      description:@"Record permissions were denied by the user. "];
      return NO;
    }

    case AVAudioSessionRecordPermissionGranted: {
      [self startTappingMicrophoneWithError:error];
      return YES;
    }

    case AVAudioSessionRecordPermissionUndetermined: {
      [TFLUtils
          createCustomError:error
                   withCode:TFLAudioErrorCodeRecordPermissionUndeterminedError
                description:@"Record permissions are undertermined. Yo must use AVAudioSession's "
                            @"requestRecordPermission() to request audio record permission from "
                            @"the user. If record permissions are granted, you can call this "
                            @"method in the completion handler of requestRecordPermission()."];
      return NO;
    }
  }
}

- (void)stop {
  [[_audioEngine inputNode] removeTapOnBus:0];
  [_audioEngine stop];
}

- (nullable TFLFloatBuffer *)readAtOffset:(NSUInteger)offset
                                 withSize:(NSUInteger)size
                                    error:(NSError *_Nullable *)error {
  __block TFLFloatBuffer *bufferToReturn = nil;
  __block NSError *readError = nil;

  dispatch_sync(_conversionQueue, ^{
    if (globalError) {
      [TFLUtils createCustomError:&readError
                         withCode:TFLAudioErrorCodeAudioProcessingError
                      description:@"Some error occured during audio processing."];
    } else if (offset + size > [_ringBuffer size]) {
      [TFLUtils createCustomError:&readError
                         withCode:TFLAudioErrorCodeInvalidArgumentError
                      description:@"Index out of bounds: offset + size should be <= to the size of "
                                  @"TFLAudioRecord's internal buffer. "];
    } else {
      bufferToReturn = [_ringBuffer floatBufferWithOffset:offset size:size];
    }
  });

  if (error) *error = readError;

  return bufferToReturn;
}

@end
