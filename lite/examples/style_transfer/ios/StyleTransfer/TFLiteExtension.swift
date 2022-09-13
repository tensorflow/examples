// Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
  /// the alpha component. This function assumes a batch size of one and three channels per image.
  /// Changing these parameters in the TF Lite model will cause conflicts.
  ///
  /// - Parameters
  ///   - size: Size to scale the image to (i.e. image size used while training the model).
  ///   - isQuantized: Whether the model is quantized (i.e. fixed point values rather than floating
  ///       point values).
  /// - Returns: The scaled image as data or `nil` if the image could not be scaled.
  func scaledData(with size: CGSize, isQuantized: Bool) -> Data? {
    guard let cgImage = self.cgImage else { return nil }
    return UIImage.normalizedData(from: cgImage, resizingTo: size, quantum: Float32.self)
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
      transform = transform.translatedBy(x: size.width, y: 0)
      transform = transform.scaledBy(x: -1, y: 1)
      break
    case .leftMirrored, .rightMirrored:
      transform = transform.translatedBy(x: size.height, y: 0)
      transform = transform.scaledBy(x: -1, y: 1)
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
    return self.pngData() ?? self.jpegData(compressionQuality: Constant.jpegCompressionQuality)
  }

  static func normalizeImage(_ image: CGImage, resizingTo size: CGSize) -> CGImage? {
    // The TF Lite model expects images in the RGB color space.
    // Device-specific RGB color spaces should have the same number of colors as the standard
    // RGB color space so we probably don't have to redraw them.
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    let cgImageSize = CGSize(width: image.width, height: image.height)
    if cgImageSize == size,
      (image.colorSpace?.name == colorSpace.name || image.colorSpace?.name == CGColorSpace.sRGB) {
      // Return the image if it is in the right format
      // to save a redraw operation.
      return image
    }
    let bitmapInfo = CGBitmapInfo(
      rawValue: CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.premultipliedLast.rawValue
    )
    let width = Int(size.width)
    let scaledBytesPerRow = (image.bytesPerRow / image.width) * width
    guard
      let context = CGContext(
        data: nil,
        width: width,
        height: Int(size.height),
        bitsPerComponent: image.bitsPerComponent,
        bytesPerRow: scaledBytesPerRow,
        space: colorSpace,
        bitmapInfo: bitmapInfo.rawValue)
    else {
      return nil
    }
    context.draw(image, in: CGRect(origin: .zero, size: size))
    return context.makeImage()
  }

  static func normalizedData<T>(from image: CGImage,
                                resizingTo size: CGSize,
                                quantum: T.Type) -> Data? where T: FloatingPoint {
    guard let normalizedImage = normalizeImage(image, resizingTo: size) else {
      return nil
    }
    guard let data = normalizedImage.dataProvider?.data as Data? else { return nil }
    // TF Lite expects an array of pixels in the form of floats normalized between 0 and 1.
    var floatArray: [T]

    // A full list of pixel formats is listed in this document under Table 2-1:
    // https://developer.apple.com/library/archive/documentation/GraphicsImaging/Conceptual/drawingwithquartz2d/dq_context/dq_context.html#//apple_ref/doc/uid/TP30001066-CH203-CJBHBFFE
    // This code only handles pixel formats supported on iOS in the RGB color space.
    // If you're targeting macOS or macOS via Catalyst, you should support the macOS
    // pixel formats as well.
    switch normalizedImage.bitsPerPixel {

    // 16-bit pixel with no alpha channel. On iOS, this must have 5 bits per channel and
    // no alpha channel. The most significant bits are skipped.
    case 16:
      guard normalizedImage.bitsPerComponent == 5 else { return nil }
      guard normalizedImage.alphaInfo.rawValue & CGImageAlphaInfo.noneSkipFirst.rawValue != 0 else {
        return nil
      }

      // If this bool is false, assume little endian byte order.
      let bigEndian: Bool = {
        // Sometimes images have both littleEndian and bigEndian flags set. In this case, use the
        // non-default endianness because it seems to work best in empirical testing.
        let hasLittleEndian = normalizedImage.bitmapInfo.contains(.byteOrder16Little)
        let hasBigEndian = normalizedImage.bitmapInfo.contains(.byteOrder16Big)
        if !(hasLittleEndian && hasBigEndian) {
          return hasBigEndian
        }
        let currentByteOrder = CFByteOrderGetCurrent()
        switch currentByteOrder {
        case Int(CFByteOrderLittleEndian.rawValue):
          return true
        case Int(CFByteOrderBigEndian.rawValue):
          return false
        case _:
          // For unknown endianness, assume little endian since it's how most
          // Apple platforms are laid out nowadays.
          return false
        }
      }()

      let initializer: (inout UnsafeMutableBufferPointer<T>, inout Int) -> () =
      { bufferPointer, initializedCount in
        let redMask: UInt16   = UInt16(0b0111110000000000)
        let greenMask: UInt16 = UInt16(0b0000001111100000)
        let blueMask: UInt16  = UInt16(0b0000000000011111)

        for byteIndex in stride(from: 0, to: data.count, by: 2) {
          // pixels are two bytes wide
          let pixelRange = byteIndex ..< byteIndex + 2
          let pixelData = data[pixelRange]
          let rawPixel = pixelData.withUnsafeBytes { $0.load(as: UInt16.self) }
          let pixel: UInt16
          if bigEndian {
            pixel = rawPixel.bigEndian
          } else {
            pixel = rawPixel.littleEndian
          }
          let redChannel   = ((pixel & redMask) &>> 10)
          let greenChannel = ((pixel & greenMask) &>> 5)
          let blueChannel  = ((pixel & blueMask) &>> 0)

          let maximumChannelValue = T(31) // 2 ^ 5 - 1
          let red   = T(redChannel) / maximumChannelValue
          let green = T(greenChannel) / maximumChannelValue
          let blue  = T(blueChannel) / maximumChannelValue

          let pixelIndex = byteIndex / 2
          let floatIndex = pixelIndex * 3
          bufferPointer[floatIndex] = red
          bufferPointer[floatIndex + 1] = green
          bufferPointer[floatIndex + 2] = blue
        }

        initializedCount = data.count / 2 * 3
      }
      floatArray = [T](unsafeUninitializedCapacity: data.count / 2 * 3,
                       initializingWith: initializer)

    // We discard the image's alpha channel before running the TF Lite model, so we can treat
    // alpha and non-alpha images identically.
    case 32:
      guard normalizedImage.bitsPerComponent == 8 else { return nil }
      let alphaFirst =
          normalizedImage.alphaInfo == CGImageAlphaInfo.noneSkipFirst ||
          normalizedImage.alphaInfo == CGImageAlphaInfo.premultipliedFirst
      let alphaLast =
          normalizedImage.alphaInfo == CGImageAlphaInfo.noneSkipLast ||
          normalizedImage.alphaInfo == CGImageAlphaInfo.premultipliedLast
      let bigEndian = normalizedImage.bitmapInfo.contains(.byteOrder32Big)
      let littleEndian = normalizedImage.bitmapInfo.contains(.byteOrder32Little)
      guard alphaFirst || alphaLast else { return nil }
      guard bigEndian || littleEndian else { return nil }

      // Iterate over channels individually. Since the order of the channels in memory
      // may vary, we cannot add channels to the float buffer we pass to TF Lite in the
      // order that they are iterated over.
      let initializer: (inout UnsafeMutableBufferPointer<T>, inout Int) -> () =
      { bufferPointer, initializedCount in
        let numberOfChannels = 4
        let alphaOffset: UInt8 = {
          if bigEndian {
            return alphaFirst ? 0 : 3
          } else {
            return alphaFirst ? 3 : 0
          }
        }()
        let redOffset: UInt8 = {
          if bigEndian {
            return alphaFirst ? 1 : 0
          } else {
            return alphaFirst ? 2 : 3
          }
        }()
        let greenOffset: UInt8 = {
          if bigEndian {
            return alphaFirst ? 2 : 1
          } else {
            return alphaFirst ? 1 : 2
          }
        }()
        let blueOffset: UInt8 = {
          if bigEndian {
            return alphaFirst ? 3 : 2
          } else {
            return alphaFirst ? 0 : 1
          }
        }()

        // Make sure we add the pixel components to the float array in the right
        // order regardless of pixel endianness.
        var rgbHolder: (red: T?, green: T?, blue: T?) = (nil, nil, nil)
        var floatIndex = 0

        func flushRGBs(_ rgbs: (red: T?, green: T?, blue: T?),
                       to array: inout UnsafeMutableBufferPointer<T>,
                       at index: Int) {
          guard let red = rgbs.red, let green = rgbs.green, let blue = rgbs.blue else { return }
          array[index] = red
          array[index + 1] = green
          array[index + 2] = blue
          floatIndex += 3
        }

        let maximumChannelValue: T = 255 // 2 ^ 8 - 1

        func normalizeChannel(_ channel: UInt8) -> T {
          return T(
            bigEndian ? channel.bigEndian : channel.littleEndian
          ) / maximumChannelValue
        }

        for component in data.enumerated() {
          switch UInt8(component.offset % numberOfChannels) {
          case alphaOffset:
            // Ignore alpha channels
            break // Breaks from the switch, not the loop

          case redOffset:
            rgbHolder.red = normalizeChannel(component.element)
          case greenOffset:
            rgbHolder.green = normalizeChannel(component.element)
          case blueOffset:
            rgbHolder.blue = normalizeChannel(component.element)

          case _:
            fatalError("Unhandled offset: \(component.offset)")
          }

          // After every 4th channel (one full pixel), write the RGBs to
          // the float buffer in the correct order.
          if component.offset % 4 == 3 {
            flushRGBs(rgbHolder, to: &bufferPointer, at: floatIndex)
            rgbHolder.red = nil; rgbHolder.green = nil; rgbHolder.blue = nil
          }
        }

        initializedCount = floatIndex
      }

      floatArray = [T](unsafeUninitializedCapacity: data.count / 4 * 3,
                       initializingWith: initializer)

    case _:
      print("Unsupported format from image: \(normalizedImage)")
      return nil
    }

    return Data(copyingBufferOf: floatArray)
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
  func toArray<T>(type: T.Type) -> [T] where T: AdditiveArithmetic {
    var array = [T](repeating: T.zero, count: self.count/MemoryLayout<T>.stride)
    _ = array.withUnsafeMutableBytes { copyBytes(to: $0) }
    return array
  }
}

// MARK: - Constants

private enum Constant {
  static let jpegCompressionQuality: CGFloat = 0.8
}
