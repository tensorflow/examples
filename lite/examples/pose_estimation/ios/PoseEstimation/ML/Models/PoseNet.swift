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

import Accelerate
import Foundation
import TensorFlowLite

/// A wrapper to run pose estimation using the PoseNet model.
final class PoseNet: PoseEstimator {

  // MARK: - Private Properties

  /// TensorFlow Lite `Interpreter` object for performing inference on a given model.
  private var interpreter: Interpreter

  /// TensorFlow lite `Tensor` of model input and output.
  private var inputTensor: Tensor
  private var heatsTensor: Tensor
  private var offsetsTensor: Tensor
  private let imageMean: Float = 127.5
  private let imageStd: Float = 127.5

  /// Model files
  private let posenetFile = FileInfo(name: "posenet", ext: "tflite")

  // MARK: - Initialization

  /// A failable initializer for `ModelDataHandler`. A new instance is created if the model is
  /// successfully loaded from the app's main bundle. Default `threadCount` is 2.
  init(threadCount: Int, delegate: Delegates) throws {
    // Construct the path to the model file.
    guard
      let modelPath = Bundle.main.path(forResource: posenetFile.name, ofType: posenetFile.ext)
    else {
      fatalError("Failed to load the model file.")
    }

    // Specify the options for the `Interpreter`.
    var options = Interpreter.Options()
    options.threadCount = threadCount

    // Specify the delegates for the `Interpreter`.
    var delegates: [Delegate]?
    switch delegate {
    case .gpu:
      delegates = [MetalDelegate()]
    case .npu:
      if let coreMLDelegate = CoreMLDelegate() {
        delegates = [coreMLDelegate]
      } else {
        delegates = nil
      }
    case .cpu:
      delegates = nil
    }
    // Create the `Interpreter`.
    interpreter = try Interpreter(modelPath: modelPath, options: options, delegates: delegates)

    // Initialize input and output `Tensor`s.
    // Allocate memory for the model's input `Tensor`s.
    try interpreter.allocateTensors()

    // Get allocated input and output `Tensor`s.
    inputTensor = try interpreter.input(at: 0)
    heatsTensor = try interpreter.output(at: 0)
    offsetsTensor = try interpreter.output(at: 1)
  }

  /// Runs PoseEstimation model with given image with given source area to destination area.
  ///
  /// - Parameters:
  ///   - on: Input image to run the model.
  ///   - from: Range of input image to run the model.
  ///   - to: Size of view to render the result.
  /// - Returns: Result of the inference and the times consumed in every steps.
  func estimateSinglePose(on pixelBuffer: CVPixelBuffer) throws -> (Person, Times) {
    // Start times of each process.
    let preprocessingStartTime: Date
    let inferenceStartTime: Date
    let postprocessingStartTime: Date

    // Processing times in seconds.
    let preprocessingTime: TimeInterval
    let inferenceTime: TimeInterval
    let postprocessingTime: TimeInterval

    preprocessingStartTime = Date()
    guard let data = preprocess(pixelBuffer) else {
      os_log("Preprocessing failed.", type: .error)
      throw PoseEstimationError.preprocessingFailed
    }
    preprocessingTime = Date().timeIntervalSince(preprocessingStartTime)

    // Run inference with the TFLite model.
    inferenceStartTime = Date()
    do {
      try interpreter.copy(data, toInputAt: 0)

      // Run inference by invoking the `Interpreter`.
      try interpreter.invoke()

      // Get the output `Tensor` to process the inference results.
      heatsTensor = try interpreter.output(at: 0)
      offsetsTensor = try interpreter.output(at: 1)

    } catch let error {
      os_log(
        "Failed to invoke the interpreter with error: %s", type: .error,
        error.localizedDescription)
      throw PoseEstimationError.inferenceFailed
    }
    inferenceTime = Date().timeIntervalSince(inferenceStartTime)

    postprocessingStartTime = Date()
    guard let result = postprocess(to: pixelBuffer.size) else {
      os_log("Postprocessing failed.", type: .error)
      throw PoseEstimationError.postProcessingFailed
    }
    postprocessingTime = Date().timeIntervalSince(postprocessingStartTime)

    let times = Times(
      preprocessing: preprocessingTime,
      inference: inferenceTime,
      postprocessing: postprocessingTime)
    return (result, times)
  }

  // MARK: - Private functions to run model
  /// Preprocesses given rectangle image to be `Data` of desired size by cropping and resizing it.
  ///
  /// - Parameters:
  ///   - of: Input image to crop and resize.
  ///   - from: Target area to be cropped and resized.
  /// - Returns: The cropped and resized image. `nil` if it can not be processed.
  private func preprocess(_ pixelBuffer: CVPixelBuffer) -> Data? {
    let sourcePixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer)
    assert(
      sourcePixelFormat == kCVPixelFormatType_32BGRA
        || sourcePixelFormat == kCVPixelFormatType_32ARGB)

    // Resize `targetSquare` of input image to `modelSize`.
    let dimensions = inputTensor.shape.dimensions
    let inputWidth = dimensions[1]
    let inputHeight = dimensions[2]
    let modelSize = CGSize(width: inputWidth, height: inputHeight)
    guard let thumbnail = pixelBuffer.resized(to: modelSize) else {
      return nil
    }
    // Remove the alpha component from the image buffer to get the initialized `Data`.
    return thumbnail.rgbData(isModelQuantized: false, imageMean: imageMean, imageStd: imageStd)
  }

  /// Postprocesses output `Tensor`s to `Result` with size of view to render the result.
  ///
  /// - Parameters:
  ///   - to: Size of view to be displayed.
  /// - Returns: Postprocessed `Result`. `nil` if it can not be processed.
  private func postprocess(to viewSize: CGSize) -> Person? {
    // MARK: Formats output tensors
    // Convert `Tensor` to `FlatArray`. As PoseEstimation is not quantized, convert them to Float
    // type `FlatArray`.
    let heats = FlatArray<Float32>(tensor: heatsTensor)
    let offsets = FlatArray<Float32>(tensor: offsetsTensor)
    let outputHeight = heats.dimensions[1]
    let outputWidth = heats.dimensions[2]
    let keyPointSize = heats.dimensions[3]
    // MARK: Find position of each key point
    // Finds the (row, col) locations of where the keypoints are most likely to be. The highest
    // `heats[0, row, col, keypoint]` value, the more likely `keypoint` being located in (`row`,
    // `col`).
    let keypointPositions = (0..<keyPointSize).map { keypoint -> (Int, Int) in
      var maxValue = heats[0, 0, 0, keypoint]
      var maxRow = 0
      var maxCol = 0
      for row in 0..<outputHeight {
        for col in 0..<outputWidth {
          if heats[0, row, col, keypoint] > maxValue {
            maxValue = heats[0, row, col, keypoint]
            maxRow = row
            maxCol = col
          }
        }
      }
      return (maxRow, maxCol)
    }

    // MARK: Calculates total confidence score
    // Calculates total confidence score of each key position.
    let totalScoreSum = keypointPositions.enumerated().reduce(0.0) { accumulator, elem -> Float32 in
      accumulator + sigmoid(heats[0, elem.element.0, elem.element.1, elem.offset])
    }
    let totalScore = totalScoreSum / Float32(keyPointSize)

    // MARK: Calculate key point position on model input
    // Calculates `KeyPoint` coordination model input image with `offsets` adjustment.
    let dimensions = inputTensor.shape.dimensions
    let inputHeight = dimensions[1]
    let inputWidth = dimensions[2]
    let coords = keypointPositions.enumerated().map { index, elem -> (y: Float32, x: Float32) in
      let (y, x) = elem
      let yCoord =
        Float32(y) / Float32(outputHeight - 1) * Float32(inputHeight)
        + offsets[0, y, x, index]
      let xCoord =
        Float32(x) / Float32(outputWidth - 1) * Float32(inputWidth)
        + offsets[0, y, x, index + keyPointSize]
      return (y: yCoord, x: xCoord)
    }

    // MARK: Transform key point position and make lines
    // Make `Result` from `keypointPosition'. Each point is adjusted to `ViewSize` to be drawn.
    var result = Person(keyPoints: [], score: totalScore)

    for (index, part) in BodyPart.allCases.enumerated() {
      let x = CGFloat(coords[index].x) * viewSize.width / CGFloat(inputWidth)
      let y = CGFloat(coords[index].y) * viewSize.height / CGFloat(inputHeight)
      let keyPoint = KeyPoint(bodyPart: part, coordinate: CGPoint(x: x, y: y))
      result.keyPoints.append(keyPoint)

    }
    return result
  }

  /// Run inference with given `Data`
  ///
  /// Parameter `from`: `Data` of input image to run model.
  private func inference(from data: Data) {
    // Copy the initialized `Data` to the input `Tensor`.
    do {
      try interpreter.copy(data, toInputAt: 0)

      // Run inference by invoking the `Interpreter`.
      try interpreter.invoke()

      // Get the output `Tensor` to process the inference results.
      heatsTensor = try interpreter.output(at: 0)
      offsetsTensor = try interpreter.output(at: 1)

    } catch let error {
      os_log(
        "Failed to invoke the interpreter with error: %s", type: .error,
        error.localizedDescription)
      return
    }
  }

  /// Returns value within [0,1].
  private func sigmoid(_ x: Float32) -> Float32 {
    return (1.0 / (1.0 + exp(-x)))
  }
}

// MARK: - Wrappers
/// Struct for handling multidimension `Data` in flat `Array`.
fileprivate struct FlatArray<Element: AdditiveArithmetic> {
  private var array: [Element]
  var dimensions: [Int]

  init(tensor: Tensor) {
    dimensions = tensor.shape.dimensions
    array = tensor.data.toArray(type: Element.self)
  }

  private func flatIndex(_ indices: [Int]) -> Int {
    guard indices.count == dimensions.count else {
      fatalError("Invalid index: got \(indices.count) index(es) for \(dimensions.count) index(es).")
    }

    var result = 0
    for (dimension, index) in zip(dimensions, indices) {
      guard dimension > index else {
        fatalError("Invalid index: \(index) is bigger than \(dimension)")
      }
      result = dimension * result + index
    }
    return result
  }

  subscript(_ index: Int...) -> Element {
    get {
      return array[flatIndex(index)]
    }
    set(newValue) {
      array[flatIndex(index)] = newValue
    }
  }
}

