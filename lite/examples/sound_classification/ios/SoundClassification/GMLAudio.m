//
//  GMLAudio.m
//  SoundClassification
//
//  Created by Prianka Kariat on 19/04/22.
//  Copyright © 2022 Tensorflow. All rights reserved.
//

#import "GMLAudio.h"

@implementation GMLAudioFormat
-(instancetype)initWithChannelCount:(NSInteger)channelCount sampleRate:(NSInteger)sampleRate {
  self = [self init];
  if (self) {
    _channelCount = channelCount;
    _sampleRate = sampleRate;
  }
  return self;
}

@end

@implementation GMLFloatBuffer

-(instancetype)initWithLength:(NSInteger)length  {
  self = [self init];
  if (self) {
    _buffer = malloc(sizeof(float) * length);
    if (!_buffer) {
      exit(-1);
    }
  }
  return self;
}

-(void)dealloc {
  free(_buffer);
}

@end


@implementation GMLAudio {
  
}

-(instancetype)initWithAudioFormat:(GMLAudioFormat *)format modelSampleCount:(NSInteger)modelSampleCount {
  self = [self init];
  if (self) {
    _audioFormat = format;
    _buffer = [[GMLFloatBuffer alloc] initWithLength:modelSampleCount];
  }
  return self;
}

//-(void)load:(GMLAudioFormat *)format modelSampleCount:(NSInteger)modelSampleCount {
//  self = [self init];
//  if (self) {
//    _audioFormat = format;
//    _buffer = [[GMLFloatBuffer alloc] initWithLength:modelSampleCount];
//  }
//  return self;
//}


@end



@implementation GMLAudioRingBuffer {
  GMLFloatBuffer *_buffer;
  NSInteger _nextIndex;
}

- (instancetype)initWithBufferLength:(NSInteger)length {
  self = [self init];
  if (self) {
    _buffer = [[GMLFloatBuffer alloc] initWithLength:length];
  }
  return self;
}


-(void)loadWithNewData:(GMLFloatBuffer *)data offset:(NSInteger)offset length:(NSInteger)length {
  
  NSInteger lengthToCopy = length;
  NSInteger newOffset = offset;
  
  if (offset + length > _buffer.length) {
    return;
  }
  
  // Length is greater than buffer length, then keep most recent data.
  if (length > _buffer.length) {
    lengthToCopy = _buffer.length;
    newOffset += length - _buffer.length;
  }
  
  // If buffer fits into the ring buffer.
  if (newOffset + lengthToCopy < _buffer.length) {
    memcpy(_buffer.buffer + _nextIndex, data.buffer + newOffset, sizeof(float) * lengthToCopy);
  }
  else {
    NSInteger firstChunkLength =_buffer.length - _nextIndex;
    memcpy(_buffer.buffer + _nextIndex, data.buffer + newOffset, sizeof(float) * firstChunkLength);
    memcpy(_buffer.buffer, data.buffer + firstChunkLength, sizeof(float) * lengthToCopy - firstChunkLength);
  }
  
  _nextIndex = (_nextIndex + lengthToCopy) % _buffer.length;
  
}



@end
