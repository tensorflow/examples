//
//  TFLAudioRecord.m
//  SoundClassification
//
//  Created by Prianka Kariat on 21/04/22.
//  Copyright © 2022 Tensorflow. All rights reserved.
//

#import "TFLAudioRecord.h"

@implementation TFLAudioRecord {
  AVAudioEngine *_audioEngine;
  GMLAudioRingBuffer *_ringBuffer;
  dispatch_queue_t delegateQueue;
}
  

-(instancetype)initWithChannelCount:(NSInteger)channelCount sampleRate:(NSInteger)sampleRate bufferSize:(NSInteger)bufferSize{
  self = [self init];
  if (self) {
    _audioFormat = [[GMLAudioFormat alloc] initWithChannelCount:channelCount sampleRate:sampleRate];
    _audioEngine = [[AVAudioEngine alloc] init];
    _bufferSize = bufferSize;
    _ringBuffer = [[GMLAudioRingBuffer alloc] initWithBufferSize:bufferSize];
    delegateQueue = dispatch_queue_create("com.gmlAudio.AudioConversionQueue",
                                           DISPATCH_QUEUE_CONCURRENT);

  }
  return self;
}


-(void)requestPermissionsWithCompletionHandler:(void(^)(NSError * _Nullable error))completionHandler {
//  [[AVAudioSession sharedInstance] requestRecordPermission:^(BOOL granted) {
//    if (granted) {
//        NSError *tapError = nil;
//      [self startTappingMicrophoneWithCompletionHandler:^(GMLFloatBuffer * _Nullable buffer, NSError * _Nullable error) {
//        <#code#>
//      }];
//      completionHandler(tapError);
//
//    }
//    else {
//      NSError *permissionError = nil;
//      [self checkAndStartTappingMicrophoneWithError:&permissionError];
//      completionHandler(permissionError);
//    }
//  }];
}
      
-(void)startTappingMicrophoneWithCompletionHandler:(void(^)(GMLFloatBuffer *_Nullable buffer, NSError * _Nullable error))completionHandler {
  AVAudioNode *inputNode = [_audioEngine inputNode];
  AVAudioFormat *format = [inputNode outputFormatForBus: 0];

  AVAudioFormat *recordingFormat = [[AVAudioFormat alloc] initWithCommonFormat:AVAudioPCMFormatFloat32 sampleRate:(double)self.audioFormat.sampleRate channels:(AVAudioChannelCount)self.audioFormat.channelCount interleaved:YES];
  
  AVAudioConverter *audioConverter = [[AVAudioConverter alloc] initFromFormat:format toFormat:recordingFormat];
  
  [inputNode installTapOnBus:0 bufferSize:1024 format:format block:^(AVAudioPCMBuffer *buffer, AVAudioTime *when) {
    dispatch_barrier_async(self->delegateQueue, ^{

    AVAudioFrameCount capacity = ceil(buffer.frameLength * recordingFormat.sampleRate / format.sampleRate);
    AVAudioPCMBuffer *pcmBuffer = [[AVAudioPCMBuffer alloc] initWithPCMFormat:recordingFormat frameCapacity:capacity * (AVAudioFrameCount)self.audioFormat.channelCount];
      
      
   AVAudioConverterInputBlock inputBlock = ^AVAudioBuffer * _Nullable(AVAudioPacketCount inNumberOfPackets, AVAudioConverterInputStatus * _Nonnull outStatus) {
     
     *outStatus = AVAudioConverterInputStatus_HaveData;
     return buffer;

   };
    
    NSError *conversionError = nil;
    AVAudioConverterOutputStatus converterStatus = [audioConverter convertToBuffer:pcmBuffer error:&conversionError withInputFromBlock:inputBlock];
    
    switch (converterStatus) {
        
      case AVAudioConverterOutputStatus_HaveData: {
        GMLFloatBuffer *floatBuffer = [[GMLFloatBuffer alloc] initWithData:pcmBuffer.floatChannelData[0] size:pcmBuffer.frameLength];
        
       
        if ( pcmBuffer.frameLength == 0) {
          NSError *checkBufferSizeAndSampleRateError = [NSError errorWithDomain:@"hello" code:121 userInfo:nil];
          completionHandler(nil, checkBufferSizeAndSampleRateError);
        }
                
        [self->_ringBuffer loadWithNewBuffer:floatBuffer offset:0 size:floatBuffer.size];
//        NSLog(@"%d",floatBuffer.size);

        GMLFloatBuffer *outFloatBuffer = [[GMLFloatBuffer alloc] initWithData:self->_ringBuffer.buffer.data size:self->_ringBuffer.buffer.size];
        completionHandler(outFloatBuffer, nil);
        break;
      }
      case AVAudioConverterOutputStatus_Error: {
        completionHandler(nil, conversionError);
        break;
      }
      default:
        completionHandler(nil, nil);
        break;
    }
    });
     
  }];
 
  NSError *engineStartError = nil;
  
  [_audioEngine prepare];
  [_audioEngine startAndReturnError:&engineStartError];
  
  if (engineStartError) {
    completionHandler(nil, engineStartError);
  }
 
}

//-(GMLFloatBuffer *)readWithData:(GMLFloatBuffer *)buffer offset:(NSInteger)offset length:(NSInteger)length {
//  
//  if (_ringBuffer.)
//}



//- (void)checkPermissionAndStartTappingMicrophoneWithCompletionHandler:(void(^)(NSError * _Nullable error))completionHandler {
//
//  switch([AVAudioSession sharedInstance].recordPermission) {
//    case AVAudioSessionRecordPermissionGranted:
//      break;
//    case AVAudioSessionRecordPermissionDenied:
//      break;
//    case AVAudioSessionRecordPermissionUndetermined: {
//      [self requestPermissions];
//      break;
//
//    }
//  }
//
//
//
//  }


- (void)checkAndStartTappingMicrophoneWithCompletionHandler:(void(^)(GMLFloatBuffer *_Nullable buffer, NSError * _Nullable error))completionHandler {
  [[AVAudioSession sharedInstance] requestRecordPermission:^(BOOL granted) {
    if (granted) {
      [self startTappingMicrophoneWithCompletionHandler:^(GMLFloatBuffer * _Nullable buffer, NSError * _Nullable error) {
        completionHandler(buffer, error);
      }];
    }
    else {
      NSError *permissionError = [NSError errorWithDomain:@"gml.imagge.errors" code:01 userInfo:nil];
      
      //      [self checkAndStartTappingMicrophoneWithError:&permissionError];
      completionHandler(nil, permissionError);
    }
  }];
  
//  [[AVAudioSession sharedInstance] requestRecordPermission:^(BOOL granted) {
//
//  }];

//  switch([AVAudioSession sharedInstance].recordPermission) {
//    case AVAudioSessionRecordPermissionGranted:{
//      [self startTappingMicrophoneWithCompletionHandler:^(GMLFloatBuffer * _Nullable buffer, NSError * _Nullable error) {
//        completionHandler(buffer, error);
//      };
//      break;
//    }
//    case AVAudioSessionRecordPermissionDenied:
//      break;
//    case AVAudioSessionRecordPermissionUndetermined: {
//      [self requestPermissionsWithCompletionHandler:^(NSError * _Nullable error) {
//
//      }];
//      break;
//    }
//    default:
//      break;
//  }
//
//  return YES;
  }

-(void)stopTappingMicrophone {
  
  [[_audioEngine inputNode] removeTapOnBus:0];
  [_audioEngine stop];
  
}


@end
