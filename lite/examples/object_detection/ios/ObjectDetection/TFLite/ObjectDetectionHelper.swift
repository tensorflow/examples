// Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

import TensorFlowLiteTaskVision

/// Stores results for a particular frame that was successfully run through the `Interpreter`.
struct Result {
  let inferenceTime: Double
  let detections: [Detection]
}

/// Information about a model file or labels file.
typealias FileInfo = (name: String, extension: String)

/// This class handles all data preprocessing and makes calls to run inference on a given frame
/// by invoking the `ObjectDetector`.
class ObjectDetectionHelper: NSObject {

  // MARK: Private properties

  /// TensorFlow Lite `ObjectDetector` object for performing object detection using a given model.
  private var detector: ObjectDetector

  private let colors = [
    UIColor.black,  // 0.0 white
    UIColor.darkGray,  // 0.333 white
    UIColor.lightGray,  // 0.667 white
    UIColor.white,  // 1.0 white
    UIColor.gray,  // 0.5 white
    UIColor.red,  // 1.0, 0.0, 0.0 RGB
    UIColor.green,  // 0.0, 1.0, 0.0 RGB
    UIColor.blue,  // 0.0, 0.0, 1.0 RGB
    UIColor.cyan,  // 0.0, 1.0, 1.0 RGB
    UIColor.yellow,  // 1.0, 1.0, 0.0 RGB
    UIColor.magenta,  // 1.0, 0.0, 1.0 RGB
    UIColor.orange,  // 1.0, 0.5, 0.0 RGB
    UIColor.purple,  // 0.5, 0.0, 0.5 RGB
    UIColor.brown,  // 0.6, 0.4, 0.2 RGB
  ]

  // MARK: - Initialization

  /// A failable initializer for `ObjectDetectionHelper`.
  ///
  /// - Parameter modelFileInfo: The TFLite model to be used.
  /// - Parameter:
  ///   - threadCount: Number of threads to be used.
  ///   - scoreThreshold: Minimum score of objects to be include in the detection result.
  ///   - maxResults: Maximum number of objects to be include in the detection result.
  /// - Returns: A new instance is created if the model is successfully loaded from the app's main
  /// bundle.
  init?(modelFileInfo: FileInfo, threadCount: Int, scoreThreshold: Float, maxResults: Int) {
    let modelFilename = modelFileInfo.name

    // Construct the path to the model file.
    guard
      let modelPath = Bundle.main.path(
        forResource: modelFileInfo.name,
        ofType: modelFileInfo.extension
      )
    else {
      print("Failed to load the model file with name: \(modelFilename).")
      return nil
    }

    // Specify the options for the `Detector`.
    let options = ObjectDetectorOptions(modelPath: modelPath)
    options.classificationOptions.scoreThreshold = scoreThreshold
    options.classificationOptions.maxResults = maxResults
    options.baseOptions.computeSettings.cpuSettings.numThreads = Int32(threadCount)
    do {
      // Create the `Detector`.
      detector = try ObjectDetector.detector(options: options)
    } catch let error {
      print("Failed to create the interpreter with error: \(error.localizedDescription)")
      return nil
    }

    super.init()
  }

  /// Detect objects from the given frame.
  ///
  /// This method handles all data preprocessing and makes calls to run inference on a given frame
  /// through the `Detector`. It then formats the inferences obtained and returns results
  /// for a successful inference.
  ///
  /// - Parameter pixelBuffer: The target frame.
  /// - Returns: The detected objects and other metadata of the inference.
  func detect(frame pixelBuffer: CVPixelBuffer) -> Result? {

    guard let mlImage = MLImage(pixelBuffer: pixelBuffer) else { return nil }
    // Run inference
    do {
      let startDate = Date()
      let detectionResult = try detector.detect(mlImage: mlImage)
      let interval = Date().timeIntervalSince(startDate) * 1000

      // Returns the detection time and detections
      return Result(inferenceTime: interval, detections: detectionResult.detections)
    } catch let error {
      print("Failed to invoke the interpreter with error: \(error.localizedDescription)")
      return nil
    }
  }
}
