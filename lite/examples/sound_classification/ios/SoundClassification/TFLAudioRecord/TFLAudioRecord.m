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
#import "TFLCommonUtils.h"
#import "TFLRingBuffer.h"
#import "TFLCommon.h"

#define SUPPORTED_CHANNEL_COUNT 2

@implementation TFLAudioRecord {
  AVAudioEngine *_audioEngine;

  /* Specifying a custom buffer size on AVAUdioEngine while tapping does not take effect. Hence we
   * are storing the returned samples in a ring buffer to acheive the desired buffer size. If
   * specified buffer size is shorter than the buffer size supported by AVAUdioEngine only the most
   * recent data of the buffer of size bufferSize will be stored by the ring buffer. */
  TFLRingBuffer *_ringBuffer;
  dispatch_queue_t _conversionQueue;
  NSError *_globalError;
}

- (nullable instancetype)initWithAudioFormat:(TFLAudioFormat *)audioFormat
                                 sampleCount:(NSUInteger)sampleCount
                                       error:(NSError *_Nullable *)error {
  self = [self init];
  if (self) {
    if (audioFormat.channelCount > SUPPORTED_CHANNEL_COUNT) {
      [TFLCommonUtils
          createCustomError:error
                   withCode:TFLSupportErrorCodeInvalidArgumentError
                description:@"The channel count provided does not match the supported "
                            @"channel count. Only upto 2 audio channels are currently supported."];
      return nil;
    }

    NSError *waitError = nil;
    [TFLCommonUtils createCustomError:&waitError
                       withCode:TFLSupportErrorCodeWaitingForNewMicInputError
                    description:@"TFLAudioRecord hasn't started receiving samples from the audio "
                                @"input source. Please wait for the input."];

    _globalError = waitError;
    _audioFormat = audioFormat;
    _audioEngine = [[AVAudioEngine alloc] init];
    _bufferSize = sampleCount * audioFormat.channelCount;
    _ringBuffer = [[TFLRingBuffer alloc] initWithBufferSize:sampleCount * audioFormat.channelCount];
    _conversionQueue =
        dispatch_queue_create("com.tflAudio.AudioConversionQueue", NULL);  // Serial Queue
  }
  return self;
}


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
                          [TFLCommonUtils createCustomError:&frameBufferProcessingError
                                             withCode:TFLSupportErrorCodeInvalidArgumentError
                                          description:@"You may have to try with a different "
                                                      @"channel count or sample rate"];
                          self->_globalError = frameBufferProcessingError;
                        } else if ((pcmBuffer.frameLength % recordingFormat.channelCount) != 0) {
                          [TFLCommonUtils
                              createCustomError:&frameBufferProcessingError
                                       withCode:TFLSupportErrorCodeInvalidArgumentError
                                    description:
                                        @"You have passed an unsupported number of channels."];
                          self->_globalError = frameBufferProcessingError;
                        } else if (![self->_ringBuffer loadBuffer:floatBuffer
                                                           offset:0
                                                             size:floatBuffer.size
                                                            error:&frameBufferProcessingError]) {
                          self->_globalError = frameBufferProcessingError;
                        } else {
                          self->_globalError = nil;
                        }
                        break;
                      }
                      case AVAudioConverterOutputStatus_Error:  // fall through
                      default: {
                        if (!conversionError) {
                          [TFLCommonUtils
                              createCustomError:&conversionError
                                       withCode:TFLSupportErrorCodeAudioProcessingError
                                    description:@"Some error occurred during audio processing"];
                        }
                        self->_globalError = conversionError;
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
      [TFLCommonUtils createCustomError:error
                         withCode:TFLSupportErrorCodeAudioRecordPermissionDeniedError
                      description:@"Record permissions were denied by the user. "];
      return NO;
    }

    case AVAudioSessionRecordPermissionGranted: {
      [self startTappingMicrophoneWithError:error];
      return YES;
    }

    case AVAudioSessionRecordPermissionUndetermined: {
      [TFLCommonUtils
          createCustomError:error
                   withCode:TFLSupportErrorCodeAudioRecordPermissionUndeterminedError
                description:@"Record permissions are undertermined. Yo must use AVAudioSession's "
                            @"requestRecordPermission() to request audio record permission from "
                            @"the user. Please read Apple's documentation for further details"
                            @"If record permissions are granted, you can call this "
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
    if (_globalError) {
      [TFLCommonUtils createCustomError:&readError
                         withCode:TFLSupportErrorCodeAudioProcessingError
                      description:@"Some error occured during audio processing."];
    } else if (offset + size > [_ringBuffer size]) {
      [TFLCommonUtils createCustomError:&readError
                         withCode:TFLSupportErrorCodeInvalidArgumentError
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
