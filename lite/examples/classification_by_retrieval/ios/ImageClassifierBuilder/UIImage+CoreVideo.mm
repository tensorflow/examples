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

#import "ios/ImageClassifierBuilder/UIImage+CoreVideo.h"

#import <CoreGraphics/CoreGraphics.h>

@implementation UIImage (CoreVideo)

- (CVPixelBufferRef)cbr_asNewPixelBuffer {
  CGImageRef CGImage = self.CGImage;

  // Create a pixel buffer.
  CGSize frameSize = CGSizeMake(CGImageGetWidth(CGImage), CGImageGetHeight(CGImage));
  NSDictionary<NSString *, NSValue *> *options = @{
    (__bridge NSString *)kCVPixelBufferCGImageCompatibilityKey : @YES,
    (__bridge NSString *)kCVPixelBufferCGBitmapContextCompatibilityKey : @YES,
  };
  CVPixelBufferRef pixelBuffer = NULL;
  CVPixelBufferCreate(kCFAllocatorDefault, frameSize.width, frameSize.height,
                      kCVPixelFormatType_32BGRA, (__bridge CFDictionaryRef)options, &pixelBuffer);

  // Access the buffer.
  CVPixelBufferLockBaseAddress(pixelBuffer, 0);
  void *pixelData = CVPixelBufferGetBaseAddress(pixelBuffer);

  // Create a context to draw into.
  CGColorSpaceRef RGBColorSpace = CGColorSpaceCreateDeviceRGB();
  CGContextRef context = CGBitmapContextCreate(
      pixelData, frameSize.width, frameSize.height, 8, CVPixelBufferGetBytesPerRow(pixelBuffer),
      RGBColorSpace, (CGBitmapInfo)kCGBitmapByteOrder32Little | kCGImageAlphaPremultipliedFirst);

  // Draw the image.
  CGContextDrawImage(context, CGRectMake(0, 0, frameSize.width, frameSize.height), CGImage);

  CGColorSpaceRelease(RGBColorSpace);
  CGContextRelease(context);
  CVPixelBufferUnlockBaseAddress(pixelBuffer, 0);
  return pixelBuffer;
}

@end
