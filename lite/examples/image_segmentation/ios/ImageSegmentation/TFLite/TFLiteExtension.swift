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
}
