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
//

import Accelerate
import Foundation
import TensorFlowLite

/// A wrapper to run pose estimation using MoveNet models
final class MoveNet: PoseEstimator {

  // MARK: - Private Properties
  // TensorFlow Lite `Interpreter` object for performing inference on a given model.
  private var interpreter: Interpreter

  // TensorFlow Lite `Tensor` of model input and output.
  private var inputTensor: Tensor
  private var outputTensor: Tensor

  // Model config
  private var torsoExpansionRatio = 1.9
  private var bodyExpandsionRatio = 1.2
  private let imageMean: Float = 0
  private let imageStd: Float = 1
  private let minCropKeyPointScore: Float32 = 0.2
  private var cropRegion: RectF?
  private var isProcessing = false

  // Model files
  private let movenetLightningFile = FileInfo(name: "movenet_lightning", ext: "tflite")
  private let movenetThunderFile = FileInfo(name: "movenet_thunder", ext: "tflite")

  // MARK: - Initialization

  /// A failable initializer for `MoveNet`. A new instance is created if the model is
  /// successfully loaded from the app's main bundle. Default `threadCount` is 4.
  init(threadCount: Int, delegate: Delegates, modelType: ModelType) throws {
    // Construct the path to the model file.
    let fileInfo: FileInfo!
    switch modelType {
    case .movenetThunder:
      fileInfo = movenetThunderFile
    case .movenetLighting:
      fileInfo = movenetLightningFile
    case .posenet: fatalError("Failed to use MoveNet")
    }
    guard
      let modelPath = Bundle.main.path(
        forResource: fileInfo.name,
        ofType: fileInfo.ext)
    else {
      fatalError("Failed to load the model file with name: \(fileInfo.name).")
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
      os_log("WARNING: MoveNet currently doesn't support NPU yet. Fall back to CPU", type: .info)
      delegates = nil
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
    outputTensor = try interpreter.output(at: 0)
  }

  /// Runs PoseEstimation model with given image with given source area to destination area.
  /// This pose detector can process only one frame at each moment.
  ///
  /// - Parameters:
  ///   - on: Input image to run the model.
  /// - Returns: Result of the inference and the times consumed in every steps.
  func estimateSinglePose(on pixelBuffer: CVPixelBuffer) throws -> (Person, Times) {
    // Check if this MoveNet instance is already processing a video frame.
    // Return an empty detection result if it's currently busy.
    guard !isProcessing else {
      throw PoseEstimationError.modelBusy
    }
    isProcessing = true
    defer {
      isProcessing = false
    }

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

    // Run inference with the TFLite model
    inferenceStartTime = Date()
    do {
      // Copy the initialized `Data` to the input `Tensor`.
      try interpreter.copy(data, toInputAt: 0)

      // Run inference by invoking the `Interpreter`.
      try interpreter.invoke()
      // Get the output `Tensor` to process the inference results.
      outputTensor = try interpreter.output(at: 0)
    } catch let error {
      os_log(
        "Failed to invoke the interpreter with error: %s", type: .error,
        error.localizedDescription)
      throw PoseEstimationError.inferenceFailed
    }
    inferenceTime = Date().timeIntervalSince(inferenceStartTime)

    postprocessingStartTime = Date()
    guard let result = postprocess(imageSize: pixelBuffer.size, modelOutput: outputTensor) else {
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

  // MARK: - Private functions to run the model
  /// Preprocesses given rectangle image to be `Data` of desired size by cropping and resizing it.
  ///
  /// - Parameters:
  ///   - of: Input image to crop and resize.
  /// - Returns: The cropped and resized image. `nil` if it can not be processed.
  private func preprocess(_ pixelBuffer: CVPixelBuffer) -> Data? {
    let sourcePixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer)
    assert(
      sourcePixelFormat == kCVPixelFormatType_32BGRA
        || sourcePixelFormat == kCVPixelFormatType_32ARGB)
    // Resize `targetSquare` of input image to `modelSize`.
    let dimensions = inputTensor.shape.dimensions
    let inputWidth = CGFloat(dimensions[1])
    let inputHeight = CGFloat(dimensions[2])
    let imageWidth = pixelBuffer.size.width
    let imageHeight = pixelBuffer.size.height

    let cropRegion = self.cropRegion ??
      initialCropRegion(imageWidth: imageWidth, imageHeight: imageHeight)
    self.cropRegion = cropRegion

    let rectF = RectF(
      left: cropRegion.left * imageWidth,
      top: cropRegion.top * imageHeight,
      right: cropRegion.right * imageWidth,
      bottom: cropRegion.bottom * imageHeight)

    // Detect region
    let modelSize = CGSize(width: inputWidth, height: inputHeight)
    guard let thumbnail = pixelBuffer.cropAndResize(fromRect: rectF.rect, toSize: modelSize) else {
      return nil
    }

    // Remove the alpha component from the image buffer to get the initialized `Data`.
    guard
      let inputData = thumbnail.rgbData(
        isModelQuantized: inputTensor.dataType == .uInt8, imageMean: imageMean, imageStd: imageStd)
    else {
      os_log("Failed to convert the image buffer to RGB data.", type: .error)
      return nil
    }

    return inputData
  }

  /// Postprocesses the output `Tensor` of TFLite model to the `Person` type
  ///
  /// - Parameters:
  ///   - imageSize: Size of the input image.
  ///   - modelOutput: Output tensor from the TFLite model.
  /// - Returns: Postprocessed `Person`. `nil` if it can not be processed.
  private func postprocess(imageSize: CGSize, modelOutput: Tensor) -> Person? {
    let imageWidth = imageSize.width
    let imageHeight = imageSize.height

    let cropRegion = self.cropRegion ??
      initialCropRegion(imageWidth: imageWidth, imageHeight: imageHeight)

    let minX: CGFloat = cropRegion.left * imageWidth
    let minY: CGFloat = cropRegion.top * imageHeight

    let output = modelOutput.data.toArray(type: Float32.self)
    let dimensions = modelOutput.shape.dimensions
    let numKeyPoints = dimensions[2]
    let inputWidth = CGFloat(inputTensor.shape.dimensions[1])
    let inputHeight = CGFloat(inputTensor.shape.dimensions[2])

    let widthRatio = (cropRegion.width * imageWidth / inputWidth)
    let heightRatio = (cropRegion.height * imageHeight / inputHeight)

    // Translate the coordinates from the model output's [0..1] back to that of
    // the input image
    var positions: [CGFloat] = []
    var totalScoreSum: Float32 = 0
    var keyPoints: [KeyPoint] = []
    for idx in 0..<numKeyPoints {
      let x = ((CGFloat(output[idx * 3 + 1]) * inputWidth) * widthRatio) + minX
      let y = ((CGFloat(output[idx * 3 + 0]) * inputHeight) * heightRatio) + minY
      positions.append(x)
      positions.append(y)
      let score = output[idx * 3 + 2]
      totalScoreSum += score
      let keyPoint = KeyPoint(
        bodyPart: BodyPart.allCases[idx], coordinate: CGPoint(x: x, y: y), score: score)
      keyPoints.append(keyPoint)
    }

    // Calculate the crop region for the subsequent frame.
    self.cropRegion = nextFrameCropRegion(
      keyPoints: keyPoints, imageWidth: imageWidth, imageHeight: imageHeight)

    // Calculates total confidence score of each key position.
    let totalScore = totalScoreSum / Float32(numKeyPoints)

    // Make `Person` from `keypoints'. Each point is adjusted to the coordinate of the input image.
    return Person(keyPoints: keyPoints, score: totalScore)
  }

  // MARK: MoveNet's smart cropping logic
  /// Determines the region to crop the image for the model to run inference on.
  /// The algorithm uses the detected joints from the previous frame to estimate
  /// the square region that encloses the full body of the target person and
  /// centers at the midpoint of two hip joints. The crop size is determined by
  /// the distances between each joints and the center point.
  /// When the model is not confident with the four torso joint predictions, the
  /// function returns a default crop which is the full image padded to square.
  private func nextFrameCropRegion(keyPoints: [KeyPoint], imageWidth: CGFloat, imageHeight: CGFloat)
    -> RectF
  {
    let targetKeyPoints = keyPoints.map { keyPoint in
        KeyPoint.init(bodyPart: keyPoint.bodyPart,
                      coordinate: CGPoint(x: keyPoint.coordinate.x, y: keyPoint.coordinate.y),
                      score: keyPoint.score)
    }
    if torsoVisible(keyPoints) {
      let centerX =
        (targetKeyPoints[BodyPart.leftHip.position].coordinate.x
          + targetKeyPoints[BodyPart.rightHip.position].coordinate.x) / 2.0
      let centerY =
        (targetKeyPoints[BodyPart.leftHip.position].coordinate.y
          + targetKeyPoints[BodyPart.rightHip.position].coordinate.y) / 2.0

      let torsoAndBodyDistances =
        determineTorsoAndBodyDistances(
          keyPoints: keyPoints, targetKeyPoints: targetKeyPoints, centerX: centerX, centerY: centerY
        )

      let list = [
        torsoAndBodyDistances.maxTorsoXDistance * CGFloat(torsoExpansionRatio),
        torsoAndBodyDistances.maxTorsoYDistance * CGFloat(torsoExpansionRatio),
        torsoAndBodyDistances.maxBodyXDistance * CGFloat(bodyExpandsionRatio),
        torsoAndBodyDistances.maxBodyYDistance * CGFloat(bodyExpandsionRatio),
      ]

      var cropLengthHalf = list.max() ?? 0.0
      let tmp: [CGFloat] = [
        centerX, CGFloat(imageWidth) - centerX, centerY, CGFloat(imageHeight) - centerY,
      ]
      cropLengthHalf = min(cropLengthHalf, tmp.max() ?? 0.0)
      let cropCornerY = centerY - cropLengthHalf
      let cropCornerX = centerX - cropLengthHalf
      if cropLengthHalf > (CGFloat(max(imageWidth, imageHeight)) / 2.0) {
        return initialCropRegion(imageWidth: imageWidth, imageHeight: imageHeight)
      } else {
        let cropLength = cropLengthHalf * 2
        return RectF(
          left: max(cropCornerX, 0) / imageWidth,
          top: max(cropCornerY, 0) / imageHeight,
          right: min((cropCornerX + cropLength) / imageWidth, 1),
          bottom: min((cropCornerY + cropLength) / imageHeight, 1))
      }
    } else {
      return initialCropRegion(imageWidth: imageWidth, imageHeight: imageHeight)
    }
  }

  /// Defines the default crop region.
  /// The function provides the initial crop region (pads the full image from both
  /// sides to make it a square image) when the algorithm cannot reliably determine
  /// the crop region from the previous frame.
  private func initialCropRegion(imageWidth: CGFloat, imageHeight: CGFloat) -> RectF {
    var xMin: CGFloat
    var yMin: CGFloat
    var width: CGFloat
    var height: CGFloat
    if imageWidth > imageHeight {
      height = 1
      width = imageHeight / imageWidth
      yMin = 0
      xMin = ((imageWidth - imageHeight) / 2.0) / imageWidth
    } else {
      width = 1
      height = imageWidth / imageHeight
      xMin = 0
      yMin = ((imageHeight - imageWidth) / 2.0) / imageHeight
    }
    return RectF(left: xMin, top: yMin, right: xMin + width, bottom: yMin + height)
  }

  /// Checks whether there are enough torso keypoints.
  /// This function checks whether the model is confident at predicting one of the
  /// shoulders/hips which is required to determine a good crop region.
  private func torsoVisible(_ keyPoints: [KeyPoint]) -> Bool {
    return
      ((keyPoints[BodyPart.leftHip.position].score > minCropKeyPointScore
      || keyPoints[BodyPart.rightHip.position].score > minCropKeyPointScore))
      && ((keyPoints[BodyPart.leftShoulder.position].score > minCropKeyPointScore
        || keyPoints[BodyPart.rightShoulder.position].score > minCropKeyPointScore))
  }

  /// Calculates the maximum distance from each keypoints to the center location.
  /// The function returns the maximum distances from the two sets of keypoints:
  /// full 17 keypoints and 4 torso keypoints. The returned information will be
  /// used to determine the crop size. See determineRectF for more detail.
  private func determineTorsoAndBodyDistances(
    keyPoints: [KeyPoint], targetKeyPoints: [KeyPoint], centerX: CGFloat, centerY: CGFloat
  ) -> TorsoAndBodyDistance {
    let torsoJoints = [
      BodyPart.leftShoulder.position,
      BodyPart.rightShoulder.position,
      BodyPart.leftHip.position,
      BodyPart.rightHip.position,
    ]

    let maxTorsoYRange = torsoJoints.lazy.map { abs(centerY - targetKeyPoints[$0].coordinate.y) }
      .max() ?? 0.0
    let maxTorsoXRange = torsoJoints.lazy.map { abs(centerX - targetKeyPoints[$0].coordinate.x) }
      .max() ?? 0.0

    let confidentKeypoints = keyPoints.lazy.filter( {$0.score < self.minCropKeyPointScore} )
    let maxBodyYRange = confidentKeypoints.map({ abs(centerY - $0.coordinate.y) }).max() ?? 0.0
    let maxBodyXRange = confidentKeypoints.map({ abs(centerX - $0.coordinate.x) }).max() ?? 0.0

    return TorsoAndBodyDistance(
      maxTorsoYDistance: maxTorsoYRange,
      maxTorsoXDistance: maxTorsoXRange,
      maxBodyYDistance: maxBodyYRange,
      maxBodyXDistance: maxBodyXRange)
  }
}

/// Size of a detected person.
struct TorsoAndBodyDistance {
  var maxTorsoYDistance: CGFloat
  var maxTorsoXDistance: CGFloat
  var maxBodyYDistance: CGFloat
  var maxBodyXDistance: CGFloat
}

/// A rectangle with calculated properties for convenient access.
struct RectF {
  var left: CGFloat
  var top: CGFloat
  var right: CGFloat
  var bottom: CGFloat
  var width: CGFloat { right - left }
  var height: CGFloat { bottom - top }

  var rect: CGRect { .init(x: left, y: top, width: width, height: height) }
}
