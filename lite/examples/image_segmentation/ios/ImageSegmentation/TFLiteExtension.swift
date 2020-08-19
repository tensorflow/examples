// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import CoreGraphics
import Foundation
import UIKit

// MARK: - UIImage

/// Extension of iOS classes that is useful for working with TensorFlow Lite computer vision models.
extension UIImage {

  /// Creates and returns a new image scaled to the given size. The image preserves its original PNG
  /// or JPEG bitmap info.
  ///
  /// - Parameter size: The size to scale the image to.
  /// - Returns: The scaled image or `nil` if image could not be resized.
  func scaledImage(with size: CGSize) -> UIImage? {
    UIGraphicsBeginImageContextWithOptions(size, false, scale)
    defer { UIGraphicsEndImageContext() }
    draw(in: CGRect(origin: .zero, size: size))
    return UIGraphicsGetImageFromCurrentImageContext()?.data.flatMap(UIImage.init)
  }

  /// Returns the data representation of the image after scaling to the given `size` and removing
  /// the alpha component.
  ///
  /// - Parameters
  ///   - size: Size to scale the image to (i.e. image size used while training the model).
  ///   - byteCount: The expected byte count for the scaled image data calculated using the values
  ///       that the model was trained on: `imageWidth * imageHeight * componentsCount * batchSize`.
  ///   - isQuantized: Whether the model is quantized (i.e. fixed point values rather than floating
  ///       point values).
  /// - Returns: The scaled image as data or `nil` if the image could not be scaled.
  func scaledData(with size: CGSize, byteCount: Int, isQuantized: Bool) -> Data? {
    guard let cgImage = self.cgImage, cgImage.width > 0, cgImage.height > 0 else { return nil }
    guard let imageData = imageData(from: cgImage, with: size) else { return nil }
    var scaledBytes = [UInt8](repeating: 0, count: byteCount)
    var index = 0
    for component in imageData.enumerated() {
      let offset = component.offset
      let isAlphaComponent = (offset % Constant.alphaComponent.baseOffset)
        == Constant.alphaComponent.moduloRemainder
      guard !isAlphaComponent else { continue }
      scaledBytes[index] = component.element
      index += 1
    }
    if isQuantized { return Data(scaledBytes) }
    let scaledFloats = scaledBytes.map { (Float32($0) - Constant.imageMean) / Constant.imageStd }
    return Data(copyingBufferOf: scaledFloats)
  }

  /// Make the same image with orientation being `.up`.
  /// - Returns:  A copy of the image with .up orientation or `nil` if the image could not be
  /// rotated.
  func transformOrientationToUp() -> UIImage? {
    // Check if the image orientation is already .up and don't need any rotation.
    guard imageOrientation != UIImage.Orientation.up else {
      // No rotation needed so return a copy of this image.
      return self.copy() as? UIImage
    }

    // Make sure that this image has an CGImage attached.
    guard let cgImage = self.cgImage else { return nil }

    // Create a CGContext to draw the rotated image to.
    guard let colorSpace = cgImage.colorSpace,
      let context = CGContext(
        data: nil,
        width: Int(size.width),
        height: Int(size.height),
        bitsPerComponent: cgImage.bitsPerComponent,
        bytesPerRow: 0,
        space: colorSpace,
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
      )
    else { return nil }

    var transform: CGAffineTransform = CGAffineTransform.identity

    // Calculate the transformation matrix that needed to bring the image orientation to .up
    switch imageOrientation {
    case .down, .downMirrored:
      transform = transform.translatedBy(x: size.width, y: size.height)
      transform = transform.rotated(by: CGFloat.pi)
      break
    case .left, .leftMirrored:
      transform = transform.translatedBy(x: size.width, y: 0)
      transform = transform.rotated(by: CGFloat.pi / 2.0)
      break
    case .right, .rightMirrored:
      transform = transform.translatedBy(x: 0, y: size.height)
      transform = transform.rotated(by: CGFloat.pi / -2.0)
      break
    case .up, .upMirrored:
      break
    @unknown default:
      break
    }

    // If the image is mirrored then flip it.
    switch imageOrientation {
    case .upMirrored, .downMirrored:
      transform.translatedBy(x: size.width, y: 0)
      transform.scaledBy(x: -1, y: 1)
      break
    case .leftMirrored, .rightMirrored:
      transform.translatedBy(x: size.height, y: 0)
      transform.scaledBy(x: -1, y: 1)
    case .up, .down, .left, .right:
      break
    @unknown default:
      break
    }

    // Apply transformation matrix to the CGContext.
    context.concatenate(transform)

    switch imageOrientation {
    case .left, .leftMirrored, .right, .rightMirrored:
      context.draw(self.cgImage!, in: CGRect(x: 0, y: 0, width: size.height, height: size.width))
    default:
      context.draw(self.cgImage!, in: CGRect(x: 0, y: 0, width: size.width, height: size.height))
      break
    }

    // Create a CGImage from the context.
    guard let newCGImage = context.makeImage() else { return nil }

    // Convert it to UIImage.
    return UIImage.init(cgImage: newCGImage, scale: 1, orientation: .up)
  }

  // MARK: - Private

  /// The PNG or JPEG data representation of the image or `nil` if the conversion failed.
  private var data: Data? {
    #if swift(>=4.2)
      return self.pngData() ?? self.jpegData(compressionQuality: Constant.jpegCompressionQuality)
    #else
      return UIImagePNGRepresentation(self)
        ?? UIImageJPEGRepresentation(self, Constant.jpegCompressionQuality)
    #endif  // swift(>=4.2)
  }

  /// Returns the image data for the given CGImage based on the given `size`.
  private func imageData(from cgImage: CGImage, with size: CGSize) -> Data? {
    let bitmapInfo = CGBitmapInfo(
      rawValue: CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.premultipliedLast.rawValue
    )
    let width = Int(size.width)
    let scaledBytesPerRow = (cgImage.bytesPerRow / cgImage.width) * width
    guard
      let context = CGContext(
        data: nil,
        width: width,
        height: Int(size.height),
        bitsPerComponent: cgImage.bitsPerComponent,
        bytesPerRow: scaledBytesPerRow,
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: bitmapInfo.rawValue)
    else {
      return nil
    }
    context.draw(cgImage, in: CGRect(origin: .zero, size: size))
    return context.makeImage()?.dataProvider?.data as Data?
  }
}

// MARK: - Data

extension Data {
  /// Creates a new buffer by copying the buffer pointer of the given array.
  ///
  /// - Warning: The given array's element type `T` must be trivial in that it can be copied bit
  ///     for bit with no indirection or reference-counting operations; otherwise, reinterpreting
  ///     data from the resulting buffer has undefined behavior.
  /// - Parameter array: An array with elements of type `T`.
  init<T>(copyingBufferOf array: [T]) {
    self = array.withUnsafeBufferPointer(Data.init)
  }

  /// Convert a Data instance to Array representation.
  func toArray<T>(type: T.Type) -> [T] where T: ExpressibleByIntegerLiteral {
    var array = [T](repeating: 0, count: self.count/MemoryLayout<T>.stride)
    _ = array.withUnsafeMutableBytes { copyBytes(to: $0) }
    return array
  }
}

// MARK: - Constants

private enum Constant {
  static let jpegCompressionQuality: CGFloat = 0.8
  static let alphaComponent = (baseOffset: 4, moduloRemainder: 3)
  static let imageMean: Float32 = 127.5
  static let imageStd: Float32 = 127.5
}
