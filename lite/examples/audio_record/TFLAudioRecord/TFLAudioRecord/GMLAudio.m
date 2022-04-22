//
//  GMLAudio.m
//  SoundClassification
//
//  Created by Prianka Kariat on 19/04/22.
//  Copyright © 2022 Tensorflow. All rights reserved.
//

#import "GMLAudio.h"

@implementation GMLAudio

-(instancetype)initWithAudioFormat:(GMLAudioFormat *)format modelSampleCount:(NSInteger)modelSampleCount {
  self = [self init];
  if (self) {
    _audioFormat = format;
    _buffer = [[GMLFloatBuffer alloc] initWithSize:modelSampleCount];
  }
  return self;
}

//-(GMLFloatBuffer *)readWithData:(GMLFloatBuffer *)buffer offset:(NSInteger)offset length:(NSInteger)length;
//
//  record.read
//}


@end







