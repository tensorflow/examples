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

/// Button for Sugessted Questions.
@IBDesignable class SuggestedQuestionButton: UIButton {
  var question: String?
  lazy var border = CAShapeLayer()
  lazy var gradient = CAGradientLayer()
  lazy var fill = CAShapeLayer()

  lazy var borderPath = CGPath(rect: CGRect(), transform: nil)

  override var bounds: CGRect {
    didSet {
      borderPath =
        UIBezierPath(
          roundedRect: self.bounds.insetBy(dx: 1, dy: 6),
          cornerRadius: 10.0
        ).cgPath
    }
  }

  required init?(coder: NSCoder) {
    super.init(coder: coder)
    addSublayers()
  }

  override init(frame: CGRect) {
    super.init(frame: frame)
    addSublayers()
  }

  convenience init(of question: String) {
    self.init(frame: CGRect.zero)
    self.question = question
  }

  override func draw(_ rect: CGRect) {
    setTitle(question, for: .normal)
    setTitleColor(.black, for: .normal)
    titleLabel?.font = .systemFont(ofSize: 11, weight: .medium)
    contentEdgeInsets = UIEdgeInsets(top: 0, left: 4, bottom: 0, right: 4)
  }

  override func layoutSubviews() {
    super.layoutSubviews()
    updateSublayers()
  }

  private func addSublayers() {
    // Add border layer to the button.
    border.lineWidth = 1.0
    border.fillColor = UIColor.clear.cgColor
    border.strokeColor = UIColor.black.cgColor

    gradient.colors = [
      UIColor.orange.cgColor,
      UIColor.orange.withAlphaComponent(0.7).cgColor,
      UIColor.yellow.cgColor,
    ]
    gradient.startPoint = CGPoint(x: 0, y: 0)
    gradient.endPoint = CGPoint(x: 1, y: 1)

    layer.addSublayer(gradient)

    // Add color fill to the button.
    fill.fillColor = UIColor.white.cgColor
    layer.addSublayer(fill)
  }

  private func updateSublayers() {
    // Update customized border along to the size of the button.
    border.frame = bounds
    border.path = borderPath

    gradient.frame = border.bounds
    gradient.mask = border

    // Update color fill along to the size of the button.
    fill.path = borderPath
  }
}
