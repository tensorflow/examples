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

#import "TFLRingBuffer.h"
#import "TFLAudioError.h"
#import "TFLUtils.h"

@implementation TFLRingBuffer {
  NSInteger nextIndex;
}

- (instancetype)initWithBufferSize:(NSInteger)size {
  self = [self init];
  if (self) {
    _buffer = [[TFLFloatBuffer alloc] initWithSize:size];
  }
  return self;
}

- (BOOL)loadBuffer:(TFLFloatBuffer *)sourceBuffer
            offset:(NSUInteger)offset
              size:(NSUInteger)size
             error:(NSError **)error {
  NSInteger sizeToCopy = size;
  NSInteger newOffset = offset;

  if (offset + size > sourceBuffer.size) {
    [TFLUtils createCustomError:error
                       withCode:TFLAudioErrorCodeInvalidArgumentError
                    description:@"offset + size exceeds the maximum size of the source buffer."];
    return NO;
  }

  // Length is greater than buffer size, then modify size and offset to
  // keep most recent data in the sourceBuffer.
  if (size >= _buffer.size) {
    sizeToCopy = _buffer.size;
    newOffset = offset + (size - _buffer.size);
  }

  // If the new nextIndex + sizeToCopy is smaller than the size of the ring buffer directly
  // copy all elements to the end of the ring buffer.
  if (nextIndex + sizeToCopy < _buffer.size) {
    memcpy(_buffer.data + nextIndex, sourceBuffer.data + newOffset, sizeof(float) * sizeToCopy);
  } else {
    //If 
    NSInteger endChunkSize = _buffer.size - nextIndex;
    memcpy(_buffer.data + nextIndex, sourceBuffer.data + newOffset, sizeof(float) * endChunkSize);

    NSInteger startChunkSize = sizeToCopy - endChunkSize;
    memcpy(_buffer.data, sourceBuffer.data + offset + endChunkSize, sizeof(float) * startChunkSize);
  }
  
  nextIndex = (nextIndex + sizeToCopy) % _buffer.size;
  
  return YES;
}

- (TFLFloatBuffer *)floatBuffer {
  TFLFloatBuffer *floatBuffer = [[TFLFloatBuffer alloc] initWithSize:_buffer.size];

  // Return buffer in correct order.
  // Buffer's beginning is marked by nextIndex.
  // Copy the first chunk starting at position nextindex to the destination buffer's
  // beginning.
  NSInteger endChunkSize = _buffer.size - nextIndex;
  memcpy(floatBuffer.data, _buffer.data + nextIndex, sizeof(float) * endChunkSize);

  // Copy the next chunk starting at position 0 until next index to the destination buffer
  // locations after the chunk size that was previously copied.
  memcpy(floatBuffer.data + endChunkSize, _buffer.data, sizeof(float) * nextIndex);

  return floatBuffer;
}

@end
