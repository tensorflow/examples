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
    [TFLCommonUtils
     createCustomError:&waitError
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

- (AVAudioPCMBuffer *)bufferFromInputBuffer:(AVAudioPCMBuffer *)pcmBuffer
                        usingAudioConverter:(AVAudioConverter *)audioConverter
                                      error:(NSError **)error {
  // Capacity of converted PCM buffer is calculated in order to maintain the same
  // latency as the input pcmBuffer.
  AVAudioFrameCount capacity =
  ceil(pcmBuffer.frameLength * audioConverter.outputFormat.sampleRate / audioConverter.inputFormat.sampleRate);
  
  AVAudioPCMBuffer *outPCMBuffer = [[AVAudioPCMBuffer alloc]
                                    initWithPCMFormat:audioConverter.outputFormat
                                    frameCapacity:capacity * (AVAudioFrameCount)audioConverter.outputFormat.channelCount];
  
  AVAudioConverterInputBlock inputBlock = ^AVAudioBuffer *_Nullable(
                                                                    AVAudioPacketCount inNumberOfPackets, AVAudioConverterInputStatus *_Nonnull outStatus) {
                                                                      *outStatus = AVAudioConverterInputStatus_HaveData;
                                                                      return pcmBuffer;
                                                                    };
  
  NSError *conversionError = nil;
  AVAudioConverterOutputStatus converterStatus = [audioConverter convertToBuffer:outPCMBuffer
                                                                           error:&conversionError
                                                              withInputFromBlock:inputBlock];
  
  switch (converterStatus) {
    case AVAudioConverterOutputStatus_HaveData: {
      return outPCMBuffer;
    }
    case AVAudioConverterOutputStatus_Error: {
      NSString *errorDescription = conversionError.localizedDescription ? conversionError.localizedDescription : @"Some error occured while processing incoming audio "
      @"frames." ;
      [TFLCommonUtils createCustomError:error
                               withCode:TFLSupportErrorCodeAudioProcessingError
                            description:errorDescription];
      break;
    }
    case AVAudioConverterOutputStatus_EndOfStream: {
      [TFLCommonUtils createCustomError:error
                               withCode:TFLSupportErrorCodeAudioProcessingError
                            description:@"Reached end of input audio stream."];
      break;
    }
    case AVAudioConverterOutputStatus_InputRanDry: {
      [TFLCommonUtils createCustomError:error
                               withCode:TFLSupportErrorCodeAudioProcessingError
                            description:@"Not enough input is available to satisy the request."];
      break;
    }
  }
  return nil;
}

- (BOOL)loadAudioPCMBuffer:(AVAudioPCMBuffer *)pcmBuffer error:(NSError **)error {
  if (pcmBuffer.frameLength == 0) {
    [TFLCommonUtils createCustomError:error
                             withCode:TFLSupportErrorCodeInvalidArgumentError
                          description:@"You may have to try with a different "
     @"channel count or sample rate"];
  } else if ((pcmBuffer.frameLength % self.audioFormat.channelCount) != 0) {
    [TFLCommonUtils createCustomError:error
                             withCode:TFLSupportErrorCodeInvalidArgumentError
                          description:@"You have passed an unsupported number of channels."];
  } else {
    TFLFloatBuffer *floatBuffer =
    [[TFLFloatBuffer alloc] initWithData:pcmBuffer.floatChannelData[0]
                                    size:pcmBuffer.frameLength];
    
    if ([self->_ringBuffer loadBuffer:floatBuffer offset:0 size:floatBuffer.size error:error]) {
      return YES;
    }
  }
  return NO;
}

- (void)startTappingMicrophoneWithError:(NSError **)error {
  AVAudioNode *inputNode = [_audioEngine inputNode];
  AVAudioFormat *format = [inputNode outputFormatForBus:0];
  
  AVAudioFormat *recordingFormat = [[AVAudioFormat alloc]
                                    initWithCommonFormat:AVAudioPCMFormatFloat32
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
      
      NSError *conversionError = nil;
      AVAudioPCMBuffer *convertedPCMBuffer = [self bufferFromInputBuffer:buffer
                                                     usingAudioConverter:audioConverter
                                                                   error:&conversionError];
      
      if (!(convertedPCMBuffer && [self loadAudioPCMBuffer:convertedPCMBuffer error:&conversionError]))  {
        self->_globalError = conversionError;
      }
      else {
        self->_globalError = nil;
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
      [TFLCommonUtils
       createCustomError:&readError
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
