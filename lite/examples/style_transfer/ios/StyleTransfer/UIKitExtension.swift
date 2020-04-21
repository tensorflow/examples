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

import UIKit

/// Helper functions for the UIImage class that is useful for this sample app.
extension UIImage {

  /// Helper function to center-crop image.
  /// - Returns: Center-cropped copy of this image
  func cropCenter() -> UIImage? {
    // Don't do anything if the image is already square.
    guard size.height != size.width else {
      return self
    }
    let isPortrait = size.height > size.width
    let smallestDimension = min(size.width, size.height)
    let croppedSize = CGSize(width: smallestDimension, height: smallestDimension)
    let croppedRect = CGRect(origin: .zero, size: croppedSize)

    UIGraphicsBeginImageContextWithOptions(croppedSize, false, scale)
    let croppingOrigin = CGPoint(
      x: isPortrait ? 0 : floor((size.width - size.height) / 2),
      y: isPortrait ? floor((size.height - size.width) / 2) : 0
    )
    guard let cgImage = cgImage?.cropping(to: CGRect(origin: croppingOrigin, size: croppedSize))
    else { return nil }
    UIImage(cgImage: cgImage).draw(in: croppedRect)
    let croppedImage = UIGraphicsGetImageFromCurrentImageContext()
    UIGraphicsEndImageContext()

    return croppedImage
  }

}
