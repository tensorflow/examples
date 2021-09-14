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

import UIKit
import os

/// Custom view to visualize the pose estimation result on top of the input image.
class OverlayView: UIImageView {

  /// Visualization configs
  private enum Config {
    static let dot = (radius: CGFloat(10), color: UIColor.orange)
    static let line = (width: CGFloat(5.0), color: UIColor.orange)
  }

  /// List of lines connecting each part to be visualized.
  private static let lines = [
    (from: BodyPart.leftWrist, to: BodyPart.leftElbow),
    (from: BodyPart.leftElbow, to: BodyPart.leftShoulder),
    (from: BodyPart.leftShoulder, to: BodyPart.rightShoulder),
    (from: BodyPart.rightShoulder, to: BodyPart.rightElbow),
    (from: BodyPart.rightElbow, to: BodyPart.rightWrist),
    (from: BodyPart.leftShoulder, to: BodyPart.leftHip),
    (from: BodyPart.leftHip, to: BodyPart.rightHip),
    (from: BodyPart.rightHip, to: BodyPart.rightShoulder),
    (from: BodyPart.leftHip, to: BodyPart.leftKnee),
    (from: BodyPart.leftKnee, to: BodyPart.leftAnkle),
    (from: BodyPart.rightHip, to: BodyPart.rightKnee),
    (from: BodyPart.rightKnee, to: BodyPart.rightAnkle),
  ]

  /// CGContext to draw the detection result.
  var context: CGContext!

  /// Draw the detected keypoints on top of the input image.
  ///
  /// - Parameters:
  ///     - image: The input image.
  ///     - person: Keypoints of the person detected (i.e. output of a pose estimation model)
  func draw(at image: UIImage, person: Person) {
    if context == nil {
      UIGraphicsBeginImageContext(image.size)
      guard let context = UIGraphicsGetCurrentContext() else {
        fatalError("set current context faild")
      }
      self.context = context
    }
    guard let strokes = strokes(from: person) else { return }
    image.draw(at: .zero)
    context.setLineWidth(Config.dot.radius)
    drawDots(at: context, dots: strokes.dots)
    drawLines(at: context, lines: strokes.lines)
    context.setStrokeColor(UIColor.blue.cgColor)
    context.strokePath()
    guard let newImage = UIGraphicsGetImageFromCurrentImageContext() else { fatalError() }
    self.image = newImage
  }

  /// Draw the dots (i.e. keypoints).
  ///
  /// - Parameters:
  ///     - context: The context to be drawn on.
  ///     - dots: The list of dots to be drawn.
  private func drawDots(at context: CGContext, dots: [CGPoint]) {
    for dot in dots {
      let dotRect = CGRect(
        x: dot.x - Config.dot.radius / 2, y: dot.y - Config.dot.radius / 2,
        width: Config.dot.radius, height: Config.dot.radius)
      let path = CGPath(
        roundedRect: dotRect, cornerWidth: Config.dot.radius, cornerHeight: Config.dot.radius,
        transform: nil)
      context.addPath(path)
    }
  }

  /// Draw the lines (i.e. conneting the keypoints).
  ///
  /// - Parameters:
  ///     - context: The context to be drawn on.
  ///     - lines: The list of lines to be drawn.
  private func drawLines(at context: CGContext, lines: [Line]) {
    for line in lines {
      context.move(to: CGPoint(x: line.from.x, y: line.from.y))
      context.addLine(to: CGPoint(x: line.to.x, y: line.to.y))
    }
  }

  /// Generate a list of strokes to draw in order to visualize the pose estimation result.
  ///
  /// - Parameters:
  ///     - person: The detected person (i.e. output of a pose estimation model).
  private func strokes(from person: Person) -> Strokes? {
    var strokes = Strokes(dots: [], lines: [])
    // MARK: Visualization of detection result
    var bodyPartToDotMap: [BodyPart: CGPoint] = [:]
    for (index, part) in BodyPart.allCases.enumerated() {
      let position = CGPoint(
        x: person.keyPoints[index].coordinate.x,
        y: person.keyPoints[index].coordinate.y)
      bodyPartToDotMap[part] = position
      strokes.dots.append(position)
    }

    do {
      try strokes.lines = OverlayView.lines.map { map throws -> Line in
        guard let from = bodyPartToDotMap[map.from] else {
          throw VisualizationError.missingBodyPart(of: map.from)
        }
        guard let to = bodyPartToDotMap[map.to] else {
          throw VisualizationError.missingBodyPart(of: map.to)
        }
        return Line(from: from, to: to)
      }
    } catch VisualizationError.missingBodyPart(let missingPart) {
      os_log("Visualization error: %s is missing.", type: .error, missingPart.rawValue)
      return nil
    } catch {
      os_log("Visualization error: %s", type: .error, error.localizedDescription)
      return nil
    }
    return strokes
  }
}

/// The strokes to be drawn in order to visualize a pose estimation result.
fileprivate struct Strokes {
  var dots: [CGPoint]
  var lines: [Line]
}

/// A straight line.
fileprivate struct Line {
  let from: CGPoint
  let to: CGPoint
}

fileprivate enum VisualizationError: Error {
  case missingBodyPart(of: BodyPart)
}
