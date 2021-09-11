// Copyright 2021 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

import Accelerate
import Foundation

extension CVPixelBuffer {

  /// Size of the buffer.
  var size: CGSize {
    return CGSize(width: CVPixelBufferGetWidth(self), height: CVPixelBufferGetHeight(self))
  }

  /// Returns thumbnail by cropping pixel buffer to biggest square and scaling the cropped image
  /// to model dimensions. This method only supports 32BGRA or 32ARGB format. It returns nil for
  /// other format.
  func resized(to size: CGSize) -> CVPixelBuffer? {
    let imageWidth = CVPixelBufferGetWidth(self)
    let imageHeight = CVPixelBufferGetHeight(self)
    let pixelBufferType = CVPixelBufferGetPixelFormatType(self)
    guard
      pixelBufferType == kCVPixelFormatType_32BGRA || pixelBufferType == kCVPixelFormatType_32ARGB
    else {
      return nil
    }

    let inputImageRowBytes = CVPixelBufferGetBytesPerRow(self)
    let imageChannels = 4
    CVPixelBufferLockBaseAddress(self, CVPixelBufferLockFlags(rawValue: 0))
    // Finds the biggest square in the pixel buffer and advances rows based on it.
    guard let inputBaseAddress = CVPixelBufferGetBaseAddress(self) else {
      return nil
    }

    // Gets vImage Buffer from input image
    var inputVImageBuffer = vImage_Buffer(
      data: inputBaseAddress, height: UInt(imageHeight), width: UInt(imageWidth),
      rowBytes: inputImageRowBytes)

    let scaledImageRowBytes = Int(size.width) * imageChannels
    guard let scaledImageBytes = malloc(Int(size.height) * scaledImageRowBytes) else {
      return nil
    }

    // Allocates a vImage buffer for scaled image.
    var scaledVImageBuffer = vImage_Buffer(
      data: scaledImageBytes, height: UInt(size.height), width: UInt(size.width),
      rowBytes: scaledImageRowBytes)

    // Performs the scale operation on input image buffer and stores it in scaled image buffer.
    let scaleError = vImageScale_ARGB8888(
      &inputVImageBuffer, &scaledVImageBuffer, nil, vImage_Flags(0))

    CVPixelBufferUnlockBaseAddress(self, CVPixelBufferLockFlags(rawValue: 0))

    guard scaleError == kvImageNoError else {
      return nil
    }

    let releaseCallBack: CVPixelBufferReleaseBytesCallback = { mutablePointer, pointer in
      if let pointer = pointer {
        free(UnsafeMutableRawPointer(mutating: pointer))
      }
    }
    var scaledPixelBuffer: CVPixelBuffer?

    // Converts the scaled vImage buffer to CVPixelBuffer
    let conversionStatus = CVPixelBufferCreateWithBytes(
      nil, Int(size.width), Int(size.height), pixelBufferType, scaledImageBytes,
      scaledImageRowBytes, releaseCallBack, nil, nil, &scaledPixelBuffer)

    guard conversionStatus == kCVReturnSuccess else {
      free(scaledImageBytes)
      return nil
    }
    return scaledPixelBuffer
  }

  /// Returns a new `CVPixelBuffer` created by taking the self area and resizing it to the
  /// specified target size. Aspect ratios of source image and destination image are expected to be
  /// same.
  ///
  /// - Parameters:
  ///   - from: Source area of image to be cropped and resized.
  ///   - to: Size to scale the image to(i.e. image size used while training the model).
  /// - Returns: The cropped and resized image of itself.
  func cropAndResize(fromRect source: CGRect, toSize size: CGSize) -> CVPixelBuffer? {
    let inputImageRowBytes = CVPixelBufferGetBytesPerRow(self)
    let imageChannels = 4
    CVPixelBufferLockBaseAddress(self, CVPixelBufferLockFlags(rawValue: 0))
    defer { CVPixelBufferUnlockBaseAddress(self, CVPixelBufferLockFlags(rawValue: 0)) }

    // Finds the address of the upper leftmost pixel of the source area.
    guard
      let inputBaseAddress = CVPixelBufferGetBaseAddress(self)?.advanced(
        by: Int(source.minY) * inputImageRowBytes + Int(source.minX) * imageChannels)
    else {
      return nil
    }

    // Crops given area as vImage Buffer.
    var croppedImage = vImage_Buffer(
      data: inputBaseAddress, height: UInt(source.height), width: UInt(source.width),
      rowBytes: inputImageRowBytes)

    let resultRowBytes = Int(size.width) * imageChannels
    guard let resultAddress = malloc(Int(size.height) * resultRowBytes) else {
      return nil
    }

    // Allocates a vacant vImage buffer for resized image.
    var resizedImage = vImage_Buffer(
      data: resultAddress,
      height: UInt(size.height), width: UInt(size.width),
      rowBytes: resultRowBytes
    )

    let error = vImageScale_ARGB8888(&croppedImage, &resizedImage, nil, vImage_Flags(0))
    CVPixelBufferUnlockBaseAddress(self, CVPixelBufferLockFlags(rawValue: 0))
    if error != kvImageNoError {
      os_log("Error scaling the image.", type: .error)
      free(resultAddress)
      return nil
    }

    let releaseCallBack: CVPixelBufferReleaseBytesCallback = { mutablePointer, pointer in
      if let pointer = pointer {
        free(UnsafeMutableRawPointer(mutating: pointer))
      }
    }

    var result: CVPixelBuffer?

    // Converts the thumbnail vImage buffer to CVPixelBuffer
    let conversionStatus = CVPixelBufferCreateWithBytes(
      nil,
      Int(size.width), Int(size.height),
      CVPixelBufferGetPixelFormatType(self),
      resultAddress,
      resultRowBytes,
      releaseCallBack,
      nil,
      nil,
      &result
    )

    guard conversionStatus == kCVReturnSuccess else {
      free(resultAddress)
      return nil
    }
    return result
  }

  /// Returns the RGB data representation of the given image buffer with the specified `byteCount`.
  ///
  /// - Parameters
  ///   - buffer: The BGRA pixel buffer to convert to RGB data.
  ///   - isModelQuantized: Whether the model is quantized (i.e. fixed point values rather than
  ///       floating point values).
  /// - Returns: The RGB data representation of the image buffer or `nil` if the buffer could not be
  ///     converted.
  func rgbData(isModelQuantized: Bool, imageMean: Float, imageStd: Float) -> Data? {
    CVPixelBufferLockBaseAddress(self, .readOnly)
    defer {
      CVPixelBufferUnlockBaseAddress(self, .readOnly)
    }
    guard let sourceData = CVPixelBufferGetBaseAddress(self) else {
      return nil
    }

    let width = CVPixelBufferGetWidth(self)
    let height = CVPixelBufferGetHeight(self)
    let sourceBytesPerRow = CVPixelBufferGetBytesPerRow(self)
    let destinationChannelCount = 3
    let destinationBytesPerRow = destinationChannelCount * width

    var sourceBuffer = vImage_Buffer(
      data: sourceData,
      height: vImagePixelCount(height),
      width: vImagePixelCount(width),
      rowBytes: sourceBytesPerRow)

    guard let destinationData = malloc(height * destinationBytesPerRow) else {
      os_log("Error: out of memory.", type: .error)
      return nil
    }

    defer {
      free(destinationData)
    }

    var destinationBuffer = vImage_Buffer(
      data: destinationData,
      height: vImagePixelCount(height),
      width: vImagePixelCount(width),
      rowBytes: destinationBytesPerRow)

    if CVPixelBufferGetPixelFormatType(self) == kCVPixelFormatType_32BGRA {
      vImageConvert_BGRA8888toRGB888(&sourceBuffer, &destinationBuffer, UInt32(kvImageNoFlags))
    } else if CVPixelBufferGetPixelFormatType(self) == kCVPixelFormatType_32ARGB {
      vImageConvert_ARGB8888toRGB888(&sourceBuffer, &destinationBuffer, UInt32(kvImageNoFlags))
    }

    let byteData = Data(bytes: destinationBuffer.data, count: destinationBuffer.rowBytes * height)
    if isModelQuantized {
      return byteData
    }

    // Not quantized, convert to floats
    let bytes = [UInt8](byteData)
    let floats = bytes.map { (Float($0) - imageMean) / imageStd }
    return Data(copyingBufferOf: floats)
  }
}
