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

#import <Foundation/Foundation.h>
#import "TFLFloatBuffer.h"

NS_ASSUME_NONNULL_BEGIN

/** An wrapper class which stores a buffer that is written in circular fashion. */
@interface TFLRingBuffer : NSObject

//@property(nonatomic, assign)NSUInteger size;
/**
 * Initializes a TFLRingBuffer by copying the array with the specified size.
 *
 * @param size Size of the ring buffer.
 *
 * @return An instance of TFLRingBuffer.
 */
- (instancetype)initWithBufferSize:(NSInteger)size;

/**
 * Loads a slice of a float array to the ring buffer. If the float array is longer than ring
 * buffer's capacity, samples with lower indices in the array will be ignored.
 *
 * @return Boolean indicating success or failure of loading operation.
 */
- (BOOL)loadBuffer:(TFLFloatBuffer *)sourceBuffer
            offset:(NSUInteger)offset
              size:(NSUInteger)size
             error:(NSError **)error;

/**
 * Returns a TFLFloatBuffer with the all the ring buffer elements in order.
 *
 * @return A TFLFloatBuffer.
 */
- (TFLFloatBuffer *)floatBuffer NS_SWIFT_NAME(floatBuffer());

/**
 * Returns a TFLFloatBuffer with the  size number of ring buffer elements in order starting at
 * offset.
 *
 * @param offset Offset in the ring buffer from which elements are to be returned.
 *
 * @param size Number of elements to be returned.
 *
 * @return A TFLFloatBuffer if offset + size is within the bounds of the ring buffer, otherwise nil.
 */
- (nullable TFLFloatBuffer *)floatBufferWithOffset:(NSUInteger)offset size:(NSUInteger)size;

/**
 * Capacity of ring buffer in number of elements.
 */
- (NSUInteger)size NS_SWIFT_NAME(size());

@end

NS_ASSUME_NONNULL_END
