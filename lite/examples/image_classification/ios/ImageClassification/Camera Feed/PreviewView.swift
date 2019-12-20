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
import AVFoundation

/**
 The camera frame is displayed on this view.
 */
class PreviewView: UIView {

  #if targetEnvironment(simulator)

  lazy private var imageView: UIImageView = {
    let imageView = UIImageView()
    imageView.contentMode = .scaleAspectFill
    imageView.translatesAutoresizingMaskIntoConstraints = false
    return imageView
  }()

  override func didMoveToSuperview() {
    super.didMoveToSuperview()
    if subviews.count == 0 {
      addSubview(imageView)
      let constraints = [
        NSLayoutConstraint(item: imageView, attribute: .top,
                           relatedBy: .equal,
                           toItem: self, attribute: .top,
                           multiplier: 1, constant: 0),
        NSLayoutConstraint(item: imageView, attribute: .leading,
                           relatedBy: .equal,
                           toItem: self, attribute: .leading,
                           multiplier: 1, constant: 0),
        NSLayoutConstraint(item: imageView, attribute: .trailing,
                           relatedBy: .equal,
                           toItem: self, attribute: .trailing,
                           multiplier: 1, constant: 0),
        NSLayoutConstraint(item: imageView, attribute: .bottom,
                           relatedBy: .equal,
                           toItem: self, attribute: .bottom,
                           multiplier: 1, constant: 0),
      ]
      addConstraints(constraints)
    }
  }

  var image: UIImage? {
    get {
      return imageView.image
    }
    set {
      imageView.image = newValue
    }
  }

  #else
  var previewLayer: AVCaptureVideoPreviewLayer {
    guard let layer = layer as? AVCaptureVideoPreviewLayer else {
      fatalError("Layer expected is of type VideoPreviewLayer")
    }
    return layer
  }

  var session: AVCaptureSession? {
    get {
      return previewLayer.session
    }
    set {
      previewLayer.session = newValue
    }
  }

  override class var layerClass: AnyClass {
    return AVCaptureVideoPreviewLayer.self
  }
  #endif
}
