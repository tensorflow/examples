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

#import "GMLRingBuffer.h"
#import "GMLAudioError.h"
#import "GMLUtils.h"

@implementation GMLRingBuffer

- (instancetype)initWithBufferSize:(NSInteger)size {
  self = [self init];
  if (self) {
    _buffer = [[GMLFloatBuffer alloc] initWithSize:size];
  }
  return self;
}

- (BOOL)loadWithBuffer:(GMLFloatBuffer *)sourceBuffer
                offset:(NSUInteger)offset
                  size:(NSUInteger)size
                 error:(NSError **)error {
  NSInteger sizeToCopy = size;
  NSInteger newOffset = offset;

  if (offset + size > sourceBuffer.size) {
    [GMLUtils createCustomError:error
                       withCode:GMLAudioErrorCodeInvalidArgumentError
                    description:@"offset + size exceeds the maximum size of the source buffer."];
    return NO;
  }

  // Length is greater than buffer length, then keep most recent data.
  if (size >= _buffer.size) {
    sizeToCopy = _buffer.size;
    newOffset = offset + (size - _buffer.size);
    memcpy(_buffer.data, sourceBuffer.data + newOffset, sizeof(float) * sizeToCopy);
  } else {
    NSInteger sizeToShiftOut = size;
    NSInteger numElementsToShift = _buffer.size - size;

    // Shift out old data from beginning of buffer.
    memcpy(_buffer.data, _buffer.data + sizeToShiftOut, sizeof(float) * numElementsToShift);

    // Insert new data to end of buffer.
    memcpy(_buffer.data + numElementsToShift, sourceBuffer.data + offset, sizeof(float) * size);
  }

  return YES;
}

@end
