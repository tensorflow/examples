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
import CoreImage

struct Result {
  let recognizedCommand: RecognizedCommand?
  let inferenceTime: Double
}

/**
 This class handles all data preprocessing and makes calls to run inference on a given audio buffer through the TfliteWrapper. It then formats the inferences obtained and averages the recognized commands by running them through RecognizeCommands.
 */
class ModelDataHandler: NSObject {

  // MARK: Constants
  let sampleRate = 16000
  private let labelOffset = 2
  private let sampleDuration = 1000
  private let minimumCount = 3
  private let averageWindowDuration: Double = 1000
  private let supressionMs: Double = 1500
  private let threshold = 0.5
  private let minTimeBetweenSamples: Double = 30

  // MARK: Variables
  private var recordingLength: Int {
    return (sampleRate * sampleDuration) / 1000
  }

  // MARK: Instance Variables
  private var labels: [String] = []
  private var buffer:[Int] = []

  // MARK: Objects that handle core functionality
  private var recognizeCommands: RecognizeCommands?
  private var tfLiteWrapper: TfliteWrapper

  // MARK: Thread Related Variables
  let threaCountLimit = 10
  var threadCount = 1

  // MARK: Initializer
  /**
   This is a failable initializer for ModelDataHandler. It successfully initializes an object of the class if the model file and labels file is found, labels can be loaded and the interpreter of TensorflowLite can be initialized successfully.
   */
  init?(modelFileName: String, labelsFileName: String, labelsFileExtension: String) {

    self.tfLiteWrapper = TfliteWrapper(modelFileName: modelFileName)
    guard self.tfLiteWrapper.setUpModelAndInterpreter() else {
      return nil
    }

    super.init()

    loadLabels(fromFileName: labelsFileName, fileExtension: labelsFileExtension)
    recognizeCommands = RecognizeCommands(
        averageWindowDuration: averageWindowDuration, detectionThreshold: 0.3,
        minimumTimeBetweenSamples: minTimeBetweenSamples, supressionTime: supressionMs,
        minimumCount: minimumCount, classLabels: labels)
  }

  /**
   This class handles all data preprocessing and makes calls to run inference on audio buffer through the TfliteWrapper. It then formats the inferences obtained and averages the results by running them through RecognizeCommands.
   */
  func runModel(onBuffer buffer: [Int16]) -> Result? {
    // Gets the input tensor for the audio buffer
    guard var outAddress = tfLiteWrapper.floatInputTensor(at: 0) else {
      return nil
    }

    // Gets input tensorf for sample rate
    guard let rateList = tfLiteWrapper.intInputTensor(at: 1) else {
      return nil
    }

    for i in 0..<buffer.count {

      outAddress.pointee = Float(buffer[i]) / 32767.0
      outAddress = outAddress.advanced(by: 1)
    }

    rateList.pointee = Int32(sampleRate)

    // Gets the time at which inference starts.
    let dateBefore = Date()
    guard tfLiteWrapper.invokeInterpreter() == true else {
      return nil
    }

    // Calculates the inference time
    let dateAfter = Date().timeIntervalSince(dateBefore) * 1000.0

    // Gets the output tensor
    guard let output = tfLiteWrapper.outputTensorData(with: 0) else {
      return nil
    }

    // Gets the formatted and averaged results.
    let command =  getResults(withScores: output)

    // Returns result.
    let result = Result(recognizedCommand: command, inferenceTime: dateAfter)

    return result
  }

  /**
   This method formats the results and runs them through Recognize Commands to average the results over a window duration.
   */
  private func getResults(withScores scores: UnsafeMutablePointer<Float>) -> RecognizedCommand? {

    var results: [Float] = []
    for i in 0..<labels.count {

      results.append(scores[i])
    }

    // Runs results through recognize commands.
    let command = recognizeCommands?.process(latestResults: results, currenTime: Date().timeIntervalSince1970 * 1000)

    // Check if command is new and the identified result is not silence or unknown.
    guard let newCommand = command,
      let index = labels.index(of: newCommand.name),
      newCommand.isNew == true,
      index >= labelOffset else {
        return nil
    }
    return newCommand
  }

  // MARK: Thread Update Methods
  /**
   Sets the number of threads on the interpreter through the TFliteWrapper
   */
  func set(numberOfThreads threadCount: Int) {

    tfLiteWrapper.setNumberOfThreads(Int32(threadCount))
    self.threadCount = Int(threadCount)
  }

  // MARK: Label Handling Methods
  /**
   Loads the labels from the labels file and stores it in an instance variable
   */
  private func loadLabels(fromFileName fileName: String, fileExtension: String) {

    guard let fileURL = Bundle.main.url(forResource: fileName, withExtension: fileExtension) else {
      fatalError("Labels file not found in bundle. Please add a labels file with name \(fileName).\(fileExtension) and try again")
    }
    do {
      let contents = try String(contentsOf: fileURL, encoding: .utf8)
      self.labels = contents.components(separatedBy: .newlines)
    }
    catch {
      fatalError("Labels file named \(fileName).\(fileExtension) cannot be read. Please add a valid labels file and try again.")
    }
  }

  /**Returns the labels other than silence and unknown for display.
   */
  func offSetLabelsForDisplay() -> [String] {

    let offSetLabels = Array(labels[labelOffset..<labels.count])
    return offSetLabels
  }
}
