// Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

#import "ios/ImageClassifierBuilder/NSData+PixelBuffer.h"

#import <Accelerate/Accelerate.h>

@implementation NSData (PixelBuffer)

+ (instancetype)cbr_RGBA8888DataFromPixelBuffer:(CVPixelBufferRef)pixelBuffer {
  switch (CVPixelBufferGetPixelFormatType(pixelBuffer)) {
    case kCVPixelFormatType_32BGRA:
      return [self cbr_RGBA8888DataFromBGRAPixelBuffer:pixelBuffer];
    case kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange:
    case kCVPixelFormatType_420YpCbCr8BiPlanarFullRange:
      return [self cbr_RGBA8888DataFromYUVPixelBuffer:pixelBuffer];
    default:
      NSLog(@"Unsupported pixel buffer format. Supported format types are BGRA, 420v and 420f");
      return nil;
  }
}

#pragma mark - Private

+ (instancetype)cbr_RGBA8888DataFromBGRAPixelBuffer:(CVPixelBufferRef)pixelBuffer {
  NSAssert(kCVPixelFormatType_32BGRA == CVPixelBufferGetPixelFormatType(pixelBuffer),
           @"Expected BGRA format.");

  // Lock the buffer address.
  CVPixelBufferLockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);

  vImage_Buffer src = {
      .data = CVPixelBufferGetBaseAddress(pixelBuffer),
      .height = CVPixelBufferGetHeight(pixelBuffer),
      .width = CVPixelBufferGetWidth(pixelBuffer),
      .rowBytes = CVPixelBufferGetBytesPerRow(pixelBuffer),
  };

  // The input buffer might be 16-byte aligned (bytes per row != 4 x width). Since not all APIs
  // support padding bytes, the destination buffer is explicitly allocated without such padding.
  vImage_Buffer dest = {
      .data = malloc(4 * src.width * src.height),
      .height = src.height,
      .width = src.width,
      .rowBytes = 4 * src.width,
  };
  if (dest.data == NULL) {
    NSLog(@"Error initializing destination pixel buffer: OOM");
    return nil;
  }

  // Source is BGRA so B = 0, G = 1, R = 2, A = 3.
  // Destination is RGBA, hence a permute map of 2, 1, 0, 3.
  const uint8_t permuteMap[4] = {2, 1, 0, 3};

  vImage_Error error = vImagePermuteChannels_ARGB8888(&src, &dest, permuteMap, kvImageNoFlags);
  if (error != kvImageNoError) {
    NSLog(@"Error permuting pixel buffer channels: %ld", error);
    return nil;
  }

  // Unlock the buffer address.
  CVPixelBufferUnlockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);

  // The NSData takes ownership of the buffer and will free it when done.
  return [NSData dataWithBytesNoCopy:dest.data length:dest.height * dest.rowBytes];
}

+ (instancetype)cbr_RGBA8888DataFromYUVPixelBuffer:(CVPixelBufferRef)pixelBuffer {
  OSType formatType = CVPixelBufferGetPixelFormatType(pixelBuffer);
  if (kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange != formatType &&
      kCVPixelFormatType_420YpCbCr8BiPlanarFullRange != formatType) {
    NSLog(@"Expected 420v or 420f format.");
    return nil;
  }
  vImage_Error error = kvImageNoError;

  // Lock the buffer address.
  CVPixelBufferLockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);

  vImage_Buffer srcYp = {
      .data = CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 0),
      .height = CVPixelBufferGetHeightOfPlane(pixelBuffer, 0),
      .width = CVPixelBufferGetWidthOfPlane(pixelBuffer, 0),
      .rowBytes = CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 0),
  };
  vImage_Buffer srcCbCr = {
      .data = CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 1),
      .height = CVPixelBufferGetHeightOfPlane(pixelBuffer, 1),
      .width = CVPixelBufferGetWidthOfPlane(pixelBuffer, 1),
      .rowBytes = CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 1),
  };

  vImage_Buffer dest;
  error = vImageBuffer_Init(&dest, srcYp.height, srcYp.width, 32 /* pixelBits */, kvImageNoFlags);
  if (error != kvImageNoError) {
    NSLog(@"Error initializing destination pixel buffer: %ld", error);
    return nil;
  }

  // Default output is ARGB so A = 0, R = 1, G = 2, B = 3.
  // Destination is RGBA, hence a permute map of 1, 2, 3, 0.
  const uint8_t permuteMap[4] = {1, 2, 3, 0};

  vImage_YpCbCrToARGB *conversionMatrix =
      (formatType == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange)
          ? [self cbr_YpCbCrFullRangeToARGBConversionMatrix]
          : [self cbr_YpCbCrVideoRangeToARGBConversionMatrix];
  error = vImageConvert_420Yp8_CbCr8ToARGB8888(&srcYp, &srcCbCr, &dest, conversionMatrix,
                                               permuteMap, 255 /* alpha */, kvImageNoFlags);
  if (error != kvImageNoError) {
    NSLog(@"Error converting YUV to RGBA: %ld", error);
    return nil;
  }

  // Unlock the buffer address.
  CVPixelBufferUnlockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);

  // The NSData takes ownership of the buffer and will free it when done.
  return [NSData dataWithBytesNoCopy:dest.data length:dest.height * dest.rowBytes];
}

+ (vImage_YpCbCrToARGB *)cbr_YpCbCrFullRangeToARGBConversionMatrix {
  static vImage_YpCbCrToARGB fullRangeConversionMatrix;
  static dispatch_once_t onceToken;
  dispatch_once(&onceToken, ^{
    // Full range 8-bit, clamped to full range.
    // https://developer.apple.com/documentation/accelerate/vimage_ypcbcrpixelrange
    vImage_YpCbCrPixelRange pixelRange = {
        .Yp_bias = 0,
        .CbCr_bias = 128,
        .YpRangeMax = 255,
        .CbCrRangeMax = 255,
        .YpMax = 255,
        .YpMin = 1,  // Wonder why not 0 in Apple doc.
        .CbCrMax = 255,
        .CbCrMin = 0,
    };
    vImage_Error error = vImageConvert_YpCbCrToARGB_GenerateConversion(
        kvImage_YpCbCrToARGBMatrix_ITU_R_709_2, &pixelRange, &fullRangeConversionMatrix,
        kvImage420Yp8_CbCr8, kvImageARGB8888, kvImageNoFlags);
    NSAssert(error == kvImageNoError, @"Error creating YUV to ARGB conversion matrix: %ld", error);
  });
  return &fullRangeConversionMatrix;
}

+ (vImage_YpCbCrToARGB *)cbr_YpCbCrVideoRangeToARGBConversionMatrix {
  static vImage_YpCbCrToARGB videoRangeConversionMatrix;
  static dispatch_once_t onceToken;
  dispatch_once(&onceToken, ^{
    // Video range 8-bit, clamped to video range.
    // https://developer.apple.com/documentation/accelerate/vimage_ypcbcrpixelrange
    vImage_YpCbCrPixelRange pixelRange = {
        .Yp_bias = 16,
        .CbCr_bias = 128,
        .YpRangeMax = 265,  // Wonder why not 235 in Apple doc.
        .CbCrRangeMax = 240,
        .YpMax = 235,
        .YpMin = 16,
        .CbCrMax = 240,
        .CbCrMin = 0,
    };
    vImage_Error error = vImageConvert_YpCbCrToARGB_GenerateConversion(
        kvImage_YpCbCrToARGBMatrix_ITU_R_709_2, &pixelRange, &videoRangeConversionMatrix,
        kvImage420Yp8_CbCr8, kvImageARGB8888, kvImageNoFlags);
    NSAssert(error == kvImageNoError, @"Error creating YUV to ARGB conversion matrix: %ld", error);
  });
  return &videoRangeConversionMatrix;
}

@end
