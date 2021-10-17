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

import TFLiteSwift_Vision

/// A result from invoking the `Interpreter`.
struct Result {
  let inferenceTime: Double
  let inferences: [Inference]
}

/// An inference from invoking the `Interpreter`.
struct Inference {
  let confidence: Float
  let label: String
}

/// Information about a model file or labels file.
typealias FileInfo = (name: String, extension: String)

/// Information about the MobileNet model.
enum MobileNet {
  static let modelInfo: FileInfo = (name: "mobilenet_quant_v1_224", extension: "tflite")
  static let labelsInfo: FileInfo = (name: "labels", extension: "txt")
}

/// This class handles all data preprocessing and makes calls to run inference on a given frame
/// by invoking the `Interpreter`. It then formats the inferences obtained and returns the top N
/// results for a successful inference.
class ModelDataHandler {

  // MARK: - Internal Properties
  /// The current thread count used by the TensorFlow Lite Interpreter.
  let threadCount: Int

  let resultCount = 3
  let threadCountLimit = 10

  // MARK: - Model Parameters
  var inputWidth: Int { interpreter.inputWidth ?? -1 }
  var inputHeight: Int { interpreter.inputHeight ?? -1 }

  // MARK: - Private Properties
  /// List of labels from the given labels file.
  private var labels: [String] = []

  /// TensorFlow Lite `Interpreter` object for performing inference on a given model.
  private var interpreter: TFLiteVisionInterpreter

  /// Information about the alpha component in RGBA data.
  private let alphaComponent = (baseOffset: 4, moduloRemainder: 3)

  // MARK: - Initialization
  /// A failable initializer for `ModelDataHandler`. A new instance is created if the model and
  /// labels files are successfully loaded from the app's main bundle. Default `threadCount` is 1.
  init?(modelFileInfo: FileInfo, labelsFileInfo: FileInfo, threadCount: Int = 1) {
    let modelFilename = modelFileInfo.name

    // Specify the options for the `Interpreter`.
    self.threadCount = threadCount
    let options = TFLiteVisionInterpreter.Options(
      modelName: modelFilename,
      threadCount: threadCount,
      normalization: .scaled(from: 0.0, to: 1.0)
    )

    do {
      // Create the `Interpreter`.
      interpreter = try TFLiteVisionInterpreter(options: options)
    } catch let error {
      print("Failed to create the interpreter with error: \(error.localizedDescription)")
      return nil
    }
    // Load the classes listed in the labels file.
    loadLabels(fileInfo: labelsFileInfo)
  }

  // MARK: - Internal Methods
  /// Performs image preprocessing, invokes the `Interpreter`, and processes the inference results.
  func runModel(onFrame pixelBuffer: CVPixelBuffer) -> Result? {

    // Run inference by invoking the `Interpreter`.
    
    let interval: TimeInterval
    let results: [Float]
    do {
      let startDate = Date()
      // Get the output `TFLiteFlatArray` to process the inference results.
      results = try interpreter.inference(with: pixelBuffer).first?.array ?? []
      
      interval = Date().timeIntervalSince(startDate) * 1000
    } catch let error {
      print("Failed to invoke the interpreter with error: \(error.localizedDescription)")
      return nil
    }

    // Process the results.
    let topNInferences = getTopN(results: results)

    // Return the inference time and inference results.
    return Result(inferenceTime: interval, inferences: topNInferences)
  }

  // MARK: - Private Methods

  /// Returns the top N inference results sorted in descending order.
  private func getTopN(results: [Float]) -> [Inference] {
    // Create a zipped array of tuples [(labelIndex: Int, confidence: Float)].
    let zippedResults = zip(labels.indices, results)

    // Sort the zipped results by confidence value in descending order.
    let sortedResults = zippedResults.sorted { $0.1 > $1.1 }.prefix(resultCount)

    // Return the `Inference` results.
    return sortedResults.map { result in Inference(confidence: result.1, label: labels[result.0]) }
  }

  /// Loads the labels from the labels file and stores them in the `labels` property.
  private func loadLabels(fileInfo: FileInfo) {
    let filename = fileInfo.name
    let fileExtension = fileInfo.extension
    guard let fileURL = Bundle.main.url(forResource: filename, withExtension: fileExtension) else {
      fatalError("Labels file not found in bundle. Please add a labels file with name " +
                   "\(filename).\(fileExtension) and try again.")
    }
    do {
      let contents = try String(contentsOf: fileURL, encoding: .utf8)
      labels = contents.components(separatedBy: .newlines)
    } catch {
      fatalError("Labels file named \(filename).\(fileExtension) cannot be read. Please add a " +
                   "valid labels file and try again.")
    }
  }
}
