//
//  GMLAudio.m
//  SoundClassification
//
//  Created by Prianka Kariat on 19/04/22.
//  Copyright © 2022 Tensorflow. All rights reserved.
//

#import "GMLAudio.h"

@implementation GMLAudio



@end


@implementation GMLAudioRingBuffer {
  float *buffer;
  NSInteger size;
}

-(void)loadWithNewData:(GMLFloatBuffer *)buffer offset:(NSInteger)offset length:(NSInteger)length {
  
  if (offset + length > buffer.length) {
    return;
  }
  
  if (offset + length > buffer.length) {
    return;
  }
  
  
}



@end
