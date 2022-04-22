//
//  GMLRingBuffer.m
//  TFLAudioRecord
//
//  Created by Prianka Kariat on 22/04/22.
//

#import "GMLRingBuffer.h"

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

-(instancetype)initWithData:(float * _Nullable)data size:(NSInteger)size  {
  self = [self init];
  if (self) {
    _size = size;
    _data = malloc(sizeof(float) * size);
    if (!_data) {
      exit(-1);
    }
    if (data) {
      memcpy(_data, data, sizeof(float) * size);
    }
  }
  return self;
}

-(instancetype)initWithSize:(NSInteger)size  {
  self = [self initWithData:NULL size:size];
  if (self) {
    _size = size;
    _data = calloc(size, sizeof(float));
    if (!_data) {
      exit(-1);
    }
  }
  return self;
}

-(void)dealloc {
  free(_data);
}

@end

@implementation GMLAudioRingBuffer 

- (instancetype)initWithBufferSize:(NSInteger)size {
  self = [self init];
  if (self) {
    _buffer = [[GMLFloatBuffer alloc] initWithSize:size];
  }
  return self;
}


-(void)loadWithNewBuffer:(GMLFloatBuffer *)inBuffer offset:(NSInteger)offset size:(NSInteger)size {
  
  NSInteger sizeToCopy = size;
  NSInteger newOffset = offset;
  
  if (offset + size > inBuffer.size) {
    return;
  }
  
  // Length is greater than buffer length, then keep most recent data.
  if (size >= _buffer.size) {
    sizeToCopy = _buffer.size;
    newOffset += size - _buffer.size;
    memcpy(_buffer.data , inBuffer.data + newOffset, sizeof(float) * sizeToCopy);
  }
  else {
    // Shift out old data from beginning of buffer.
    NSInteger sizeToShift = _buffer.size - inBuffer.size;
    
    // Insert new data to end of buffer.
    memcpy(_buffer.data, _buffer.data + sizeToShift, sizeof(float) * sizeToShift);
    memcpy(_buffer.data + sizeToShift, inBuffer.data, sizeof(float) * inBuffer.size);
  }
//
//  // If buffer fits into the ring buffer.
//  if (newOffset + lengthToCopy < _buffer.length) {
//    meemcpy(_buffer.buffer, )
//  }
//  else {
//    NSInteger firstChunkLength =_buffer.length - _nextIndex;
//    memcpy(_buffer.buffer + _nextIndex, data.buffer + newOffset, sizeof(float) * firstChunkLength);
//    memcpy(_buffer.buffer, data.buffer + firstChunkLength, sizeof(float) * lengthToCopy - firstChunkLength);
//  }
//
//  _nextIndex = (_nextIndex + lengthToCopy) % _buffer.length;
  
}

@end

