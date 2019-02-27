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

/** This is a view which has it's top left and right corners rounded.
 */
class CurvedView: UIView {

  let cornerRadius: CGFloat = 24.0

  override func layoutSubviews() {
    super.layoutSubviews()
    setMask()

  }

  // Sets a mask on the view to round it's corners
  func setMask() {

    let maskPath = UIBezierPath(roundedRect:self.bounds,
                                byRoundingCorners: [.topLeft, .topRight],
                                cornerRadii: CGSize(width: cornerRadius, height: cornerRadius))

    let shape = CAShapeLayer()
    shape.path = maskPath.cgPath
    self.layer.mask = shape
  }
}
