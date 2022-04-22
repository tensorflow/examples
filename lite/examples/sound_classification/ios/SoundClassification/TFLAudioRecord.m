//
//  TFLAudioRecord.m
//  SoundClassification
//
//  Created by Prianka Kariat on 21/04/22.
//  Copyright © 2022 Tensorflow. All rights reserved.
//

#import "TFLAudioRecord.h"
@import AVFoundation;

@implementation TFLAudioRecord {
  AVAudioEngine *_audioEngine;
  
}

-(instancetype)initWithChannelCount:(NSInteger)channelCount sampleRate:(NSInteger)sampleRate bufferSize:(NSInteger)bufferSize{
  self = [self init];
  if (self) {
    _audioFormat = [[GMLAudioFormat alloc] initWithChannelCount:channelCount sampleRate:sampleRate];
    _bufferSize = bufferSize;
    _audioEngine = [[AVAudioEngine alloc] init];
  }
  return self;
}


-(void)checkPermissionStatus {
  switch([AVAudioSession sharedInstance].recordPermission) {
    case AVAudioSessionRecordPermissionGranted:
      break;
    case AVAudioSessionRecordPermissionDenied:
      break;
    case AVAudioSessionRecordPermissionUndetermined: {
      break;
      
    }
  }
}
      
-(void)startTappingMicrophone {
  AVAudioInputNode *input = [_audioEngine inputNode];
  AVAudioMixerNode *mixer = [[AVAudioMixerNode alloc] init];
  [_audioEngine attachNode:mixer];
//  AVAudioFormat *format = [input outputFormatForBus: 0];
  AVAudioFormat *recordingFormat = [[AVAudioFormat alloc] initWithCommonFormat:AVAudioPCMFormatFloat32 sampleRate:(double)self.audioFormat.sampleRate channels:(AVAudioChannelCount)self.audioFormat.channelCount interleaved:YES];
  
  [input installTapOnBus: 0 bufferSize: (AVAudioFrameCount)self.bufferSize format: format block: ^(AVAudioPCMBuffer *buf, AVAudioTime *when) {
    NSLog(@"%f", buf.floatChannelData[0]);
  }];
  
  NSError *startError = nil;
  [_audioEngine prepare];
  [_audioEngine startAndReturnError:&startError];

}

- (void)checkPermissionAndStartTappingMicrophoneWithCompletionHandler:(void(^)(NSError * _Nullable error))completionHandler {
  
 [[AVAudioSession sharedInstance] requestRecordPermission:^(BOOL granted) {
    if (granted) {
      dispatch_async(dispatch_get_main_queue(), ^{
        [self startTappingMicrophone];
      });
    }
    else {
      
      completionHandler([NSError errorWithDomain:@"Hello" code:121 userInfo:nil]);
    }
 }];

  }


@end
