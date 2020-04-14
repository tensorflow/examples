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

/// Custom view for control panel which has a separating line at the top of it.
class ControlPanelView: UIView {
  lazy var separatingLine = CALayer()

  required init?(coder: NSCoder) {
    super.init(coder: coder)
    addSeparatingLine()
  }

  override init(frame: CGRect) {
    super.init(frame: frame)
    addSeparatingLine()
  }

  override func layoutSubviews() {
    super.layoutSubviews()
    updateSeparatingLine()
  }

  /// Add seperating line at the top of the view.
  private func addSeparatingLine() {
    separatingLine.backgroundColor = UIColor.lightGray.cgColor
    layer.addSublayer(separatingLine)
  }

  /// Update separating line.
  private func updateSeparatingLine() {
    separatingLine.frame = getSeparatingLineFrame()
  }

  /// Get border of separating line to fill the width of the view.
  private func getSeparatingLineFrame() -> CGRect {
    var width = frame.width

    let left = safeAreaInsets.right
    let right = safeAreaInsets.right
    width = width + left + right

    return CGRect(x: -left, y: 0, width: width, height: 0.7)
  }
}
