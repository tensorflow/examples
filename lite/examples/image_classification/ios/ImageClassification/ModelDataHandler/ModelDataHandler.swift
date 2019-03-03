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

// MARK: Structures That hold results
/**
 Stores inference results for a particular frame that was successfully run through the TfliteWrapper.
 */
struct Result {
  let inferenceTime: Double
  let inferences: [Inference]
}

/**
 Stores one formatted inference.
 */
struct Inference {
  let confidence: Double
  let className: String
}

/**
 This class handles all data preprocessing and makes calls to run inference on a given frame through
 the TfliteWrapper. It then formats the inferences obtained and returns the top N results for a
 successful inference.
 */
class ModelDataHandler: NSObject {

  // MARK: Paremeters on which model was trained
  let wantedInputChannels = 3
  let wantedInputWidth = 224
  let wantedInputHeight = 224

  // MARK: Constants
  let resultCount = 3
  let threadCountLimit = 10

  // MARK: Instance Variables
  var labels: [String] = []
  private var threadCount: Int32 = 1

  // MARK: TfliteWrapper
  var tfLiteWrapper: TfliteWrapper

  // MARK: Initializer
  /**
   This is a failable initializer for ModelDataHandler. It successfully initializes an object of the class if the model file and labels file is found, labels can be loaded and the interpreter of TensorflowLite can be initialized successfully.
   */
  init?(modelFileName: String, labelsFileName: String, labelsFileExtension: String) {

    // Initializes TFliteWrapper and based on the setup result of interpreter, initializes the object of this class
    self.tfLiteWrapper = TfliteWrapper(modelFileName: modelFileName)
    guard self.tfLiteWrapper.setUpModelAndInterpreter() else {
      return nil
    }

    super.init()

    tfLiteWrapper.setNumberOfThreads(threadCount)

    // Opens and loads the classes listed in labels file
    loadLabels(fromFileName: labelsFileName, fileExtension: labelsFileExtension)
  }

  // MARK: Methods for data preprocessing and post processing.
  /** Performs image preprocessing, calls the tfliteWrapper methods to feed the pixel buffer into the input tensor and  run inference
   on the pixel buffer.

   */
  func runModel(onFrame pixelBuffer: CVPixelBuffer) -> Result? {

    let sourcePixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer)
    assert(sourcePixelFormat == kCVPixelFormatType_32ARGB ||
      sourcePixelFormat == kCVPixelFormatType_32BGRA || sourcePixelFormat == kCVPixelFormatType_32RGBA)


    let imageChannels = 4
    assert(imageChannels >= wantedInputChannels)

    // Crops the image to the biggest square in the center and scales it down to model dimensions.
    guard let thumbnailPixelBuffer = pixelBuffer.centerThumbnail(ofSize: CGSize(width: wantedInputWidth, height: wantedInputHeight)) else {
      return nil
    }

    CVPixelBufferLockBaseAddress(thumbnailPixelBuffer, CVPixelBufferLockFlags(rawValue: 0))

    guard let sourceStartAddrss = CVPixelBufferGetBaseAddress(thumbnailPixelBuffer) else {
      return nil
    }

    // Obtains the input tensor to feed the pixel buffer into
    guard  let tensorInputBaseAddress = tfLiteWrapper.inputTensor(at: 0) else {
      return nil
    }

    let inputImageBaseAddress = sourceStartAddrss.assumingMemoryBound(to: UInt8.self)

    for y in 0...wantedInputHeight - 1 {
      let tensorInputRow = tensorInputBaseAddress.advanced(by: (y * wantedInputWidth * wantedInputChannels))
      let inputImageRow = inputImageBaseAddress.advanced(by: y * wantedInputWidth * imageChannels)

      for x in 0...wantedInputWidth - 1 {

        let out_pixel = tensorInputRow.advanced(by: x * wantedInputChannels)
        let in_pixel = inputImageRow.advanced(by: x * imageChannels)

        var b = 2
        for c in 0...(wantedInputChannels) - 1 {

          // Pixel values are between 0-255. Model requires the values to be between -1 and 1.
          // We are also reversing the order of pixels since the source pixel format is BGRA, but the model requires RGB format.
          out_pixel[c] = in_pixel[b]
          b = b - 1
        }
      }
    }

    // Runs inference
    let dateBefore = Date()
    guard tfLiteWrapper.invokeInterpreter() else {
      return nil
    }

    // Calculates the inference time.
    let interval = Date().timeIntervalSince(dateBefore) * 1000

    // Gets the output tensor at index 0. This is a vector that holds the confidence values of the classes detected.
    guard let outputTensor = tfLiteWrapper.outputTensor(at: 0) else {
      return nil
    }

    CVPixelBufferUnlockBaseAddress(pixelBuffer, CVPixelBufferLockFlags(rawValue: 0))

    // Formats the results
    let resultArray = getTopN(prediction: outputTensor, predictionSize: 1000, resultCount: resultCount)

    // Returns the inference time and inferences
    return Result(inferenceTime: interval, inferences: resultArray)
  }

  /**
   This method filters out all the results with confidence score < threshold and returns the top N results sorted in descending order.

   */
  func getTopN(prediction: UnsafeMutablePointer<UInt8>, predictionSize:Int, resultCount: Int) -> [Inference] {

    var resultsArray: [Inference] = []

    // Filters out results with confidence score < threshold and creates and Inference object for each class detected.
    for i in 0...predictionSize - 1 {
      let value = Double(prediction[i]) / 255.0

      guard i < labels.count else {
        continue
      }

      let inference = Inference(confidence: value, className: labels[i])
      resultsArray.append(inference)
    }

    // Sorts Inferences in descending order and returns the top resultCount inferences.
    resultsArray.sort { (first, second) -> Bool in
      return first.confidence  > second.confidence
    }

    guard resultsArray.count > resultCount else {
      return resultsArray
    }
    let finalArray = resultsArray[0..<resultCount]

    return Array(finalArray)
  }

  // MARK: Thread Update Methods
  func numberOfThreads() -> Int32 {
    return threadCount
  }

  /**
   Sets the number of threads on the interpreter through the TFliteWrapper
   */
  func set(numberOfThreads threadCount: Int32) {

    tfLiteWrapper.setNumberOfThreads(threadCount)
    self.threadCount = threadCount
  }

  /**
   Loads the labels from the labels file and stores it in an instance variable
   */
  private func loadLabels(fromFileName fileName: String, fileExtension: String) {

    guard let fileURL = Bundle.main.url(forResource: fileName, withExtension: fileExtension) else {
      fatalError("Labels file not found in bundle. Please add a labels file with name \(fileName).\(fileExtension) and try again.")
    }
    do {
      let contents = try String(contentsOf: fileURL, encoding: .utf8)
      self.labels = contents.components(separatedBy: .newlines)
    }
    catch {
      fatalError("Labels file named \(fileName).\(fileExtension) cannot be read. Please add a valid labels file and try again.")
    }
  }
}
