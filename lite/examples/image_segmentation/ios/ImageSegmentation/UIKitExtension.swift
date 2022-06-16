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

import UIKit

/// Helper functions for the UIImage class that is useful for this sample app.
extension UIImage {

  /// Helper function to center-crop image.
  ///
  /// - Returns: Center-cropped copy of this image
  func cropCenter() -> UIImage? {
    let isPortrait = size.height > size.width
    let isLandscape = size.width > size.height
    let breadth = min(size.width, size.height)
    let breadthSize = CGSize(width: breadth, height: breadth)
    let breadthRect = CGRect(origin: .zero, size: breadthSize)

    UIGraphicsBeginImageContextWithOptions(breadthSize, false, scale)
    let croppingOrigin = CGPoint(
      x: isLandscape ? floor((size.width - size.height) / 2) : 0,
      y: isPortrait ? floor((size.height - size.width) / 2) : 0
    )
    guard let cgImage = cgImage?.cropping(to: CGRect(origin: croppingOrigin, size: breadthSize))
    else { return nil }
    UIImage(cgImage: cgImage).draw(in: breadthRect)
    let croppedImage = UIGraphicsGetImageFromCurrentImageContext()
    UIGraphicsEndImageContext()

    return croppedImage
  }

  /// Create an `UIImage` from the given pixel array.
  ///
  /// - Parameters
  ///   - pixels: The pixel array to create an image from.
  ///   - size: The target image's size.
  ///
  /// - Returns: The `UIImage` object or `nil` if the image could not be drawn.
  static func fromSRGBColorArray(pixels: [UInt32], size: CGSize) -> UIImage? {
    guard size.width > 0 && size.height > 0 else {
      print("ERROR: The target image size must be positive.")
      return nil
    }

    // Fails if the size of the target doesn't match with the total pixels in the SRGB array.
    guard pixels.count == Int(size.width * size.height) else {
      print(
        "ERROR: The size of the target image (\(size)) doesn't match with the total number of ",
        "pixels (\(pixels.count)) in the SRGB array."
      )
      return nil
    }

    // Make a mutable copy.
    var data = pixels

    // Convert array of pixels to a `CGImage` instance.
    let cgImage = data.withUnsafeMutableBytes { (ptr) -> CGImage in
      let ctx = CGContext(
        data: ptr.baseAddress,
        width: Int(size.width),
        height: Int(size.height),
        bitsPerComponent: 8,
        bytesPerRow: MemoryLayout<UInt32>.size * Int(size.width),
        space: CGColorSpace(name: CGColorSpace.sRGB)!,
        bitmapInfo: CGBitmapInfo.byteOrder32Little.rawValue
          + CGImageAlphaInfo.premultipliedFirst.rawValue
      )!
      return ctx.makeImage()!
    }

    // Convert the `CGImage` instance to an `UIImage` instance.
    return UIImage(cgImage: cgImage)
  }
}

/// Helper functions for the `UIKit` class that is useful for this sample app.
extension UIColor {

  // Check if the color is light or dark, as defined by the injected lightness threshold.
  // A `nil` value is returned if the lightness couldn't be determined.
  func isLight(threshold: Float = 0.5) -> Bool? {
    let originalCGColor = self.cgColor

    // Convert the color to the RGB colorspace as some color such as `UIColor.white` and `.black`
    // are grayscale.
    let RGBCGColor = originalCGColor.converted(
      to: CGColorSpaceCreateDeviceRGB(), intent: .defaultIntent, options: nil)

    guard let components = RGBCGColor?.components else { return nil }
    guard components.count >= 3 else { return nil }

    // Calculate color brightness according to Digital ITU BT.601.
    let brightness = Float(
      ((components[0] * 299) + (components[1] * 587) + (components[2] * 114)) / 1000
    )

    return (brightness > threshold)
  }
}
