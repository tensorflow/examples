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
  let rect: CGRect
  let displayColor: UIColor
}

/**
 This class handles all data preprocessing and makes calls to run inference on a given frame through the TfliteWrapper. It then formats the inferences obtained and returns the top N results for a successful inference.
 */
class ModelDataHandler: NSObject {

  // MARK: Paremeters on which model was trained
  let wantedInputChannels = 3
  let wantedInputWidth = 300
  let wantedInputHeight = 300

  // MARK: Constants
  let threaCountLimit = 10
  let threshold: Double = 0.5
  private let colorStrideValue = 10
  private let colors = [UIColor.red, UIColor(displayP3Red: 90.0/255.0, green: 200.0/255.0, blue: 250.0/255.0, alpha: 1.0), UIColor.green, UIColor.orange, UIColor.blue, UIColor.purple, UIColor.magenta, UIColor.yellow, UIColor.cyan, UIColor.brown]


  // MARK: Instance Variables
  var labels: [String] = []

  // MARK: TfliteWrapper
  var tfLiteWrapper: TfliteWrapper

  var threadCount = 1

  // MARK: Initializer
  /**
   This is a failable initializer for ModelDataHandler. It successfully initializes an object of the class if the model file and labels file is found, labels can be loaded and the interpreter of TensorflowLite can be initialized successfully.
   */
  init?(modelFileName: String, labelsFileName: String, labelsFileExtension: String) {

    // Initializes TFliteWrapper and based on the setup result of interpreter, initializes the object of this class.
    self.tfLiteWrapper = TfliteWrapper(modelFileName: modelFileName)
    guard self.tfLiteWrapper.setUpModelAndInterpreter() else {
      return nil
    }

    super.init()

    // Opens and loads the classes listed in labels file.
    loadLabels(fromFileName: labelsFileName, fileExtension: labelsFileExtension)
  }

  /**
   This class handles all data preprocessing and makes calls to run inference on a given frame through the TfliteWrapper. It then formats the inferences obtained and returns the top N results for a successful inference.
   */
  func runModel(onFrame pixelBuffer: CVPixelBuffer) -> Result? {

    let imageWidth = CVPixelBufferGetWidth(pixelBuffer)
    let imageHeight = CVPixelBufferGetHeight(pixelBuffer)
    let sourcePixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer)
    assert(sourcePixelFormat == kCVPixelFormatType_32ARGB ||
      sourcePixelFormat == kCVPixelFormatType_32BGRA || sourcePixelFormat == kCVPixelFormatType_32RGBA)


    let imageChannels = 4
    assert(imageChannels >= wantedInputChannels)

    // Crops the image to the biggest square in the center and scales it down to model dimensions.
    guard let thumbnailPixelBuffer = pixelBuffer.resized(toSize: CGSize(width: wantedInputWidth, height: wantedInputHeight)) else {
      return nil
    }

    CVPixelBufferLockBaseAddress(thumbnailPixelBuffer, CVPixelBufferLockFlags(rawValue: 0))

    guard let sourceStartAddrss = CVPixelBufferGetBaseAddress(thumbnailPixelBuffer) else {
      return nil
    }

    // Obtains the input tensor to feed the pixel buffer into
    guard let inputTensorBaseAddress = tfLiteWrapper.inputTensort(at: 0) else {
      return nil
    }

    let inputImageBaseAddress = sourceStartAddrss.assumingMemoryBound(to: UInt8.self)

    for y in 0...wantedInputHeight - 1 {
      let tensorInputRow = inputTensorBaseAddress.advanced(by: (y * wantedInputWidth * wantedInputChannels))
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
    guard tfLiteWrapper.invokeInterpreter() == true else {
      return nil
    }

    // Calculates the inference time.
    let inferenceTime = Date().timeIntervalSince(dateBefore) * 1000

    // Gets the output tensor at indices 0, 1, 2, 3 for respectively bounding boxes, detected output class indices, confidence scores and number of detected classes.
    guard let boundingBox = tfLiteWrapper.outputTensor(at: 0), let outputClasses = tfLiteWrapper.outputTensor(at: 1), let outputScores = tfLiteWrapper.outputTensor(at: 2), let outputCount = tfLiteWrapper.outputTensor(at: 3) else {

      return nil
    }

    let totalOutputCount = outputCount.pointee

    // Formats the results
    let resultArray = formatResults(withboundingBox: boundingBox, outputClasses: outputClasses, outputScores: outputScores, outputCount: Int(totalOutputCount), width: CGFloat(imageWidth), height: CGFloat(imageHeight))
    CVPixelBufferUnlockBaseAddress(pixelBuffer, CVPixelBufferLockFlags(rawValue: 0))

    // Returns the inference time and inferences
    let result = Result(inferenceTime: inferenceTime, inferences: resultArray)
    return result
  }

  /**
   This method filters out all the results with confidence score < threshold and returns the top N results sorted in descending order.

   */
  func formatResults(withboundingBox boundingBox: UnsafeMutablePointer<Float>, outputClasses: UnsafeMutablePointer<Float>, outputScores: UnsafeMutablePointer<Float>, outputCount: Int, width: CGFloat, height: CGFloat) -> [Inference]{

    var resultsArray: [Inference] = []
    for i in 0...outputCount - 1 {

      let score = Double(outputScores[i])

      // Filters results with confidence < threshold.
      guard score >= threshold else {
        continue
      }

      // Gets the output class names for detected classes from labels list.
      let outputClassIndex = Int(outputClasses[i])
      let outputClass = labels[outputClassIndex + 1]

      var rect: CGRect = CGRect.zero

      // Translates the detected bounding box to CGRect.
      rect.origin.y = CGFloat(boundingBox[4*i])
      rect.origin.x = CGFloat(boundingBox[4*i+1])
      rect.size.height = CGFloat(boundingBox[4*i+2]) - rect.origin.y
      rect.size.width = CGFloat(boundingBox[4*i+3]) - rect.origin.x

      // The detected corners are for model dimensions. So we scale the rect with respect to the actual image dimensions.
      let newRect = rect.applying(CGAffineTransform(scaleX: width, y: height))

      // Gets the color assigned for the class
      let colorToAssign = colorForClass(withIndex: outputClassIndex + 1)
      let inference = Inference(confidence: score, className: outputClass, rect: newRect, displayColor: colorToAssign)
      resultsArray.append(inference)
    }

    // Sort results in descending order of confidence.
    resultsArray.sort { (first, second) -> Bool in
      return first.confidence  > second.confidence
    }

    return resultsArray
  }

  // MARK: Thread Update Methods
  /**
   Sets the number of threads on the interpreter through the TFliteWrapper
   */
  func set(numberOfThreads threadCount: Int) {

    tfLiteWrapper.setNumberOfThreads(Int32(threadCount))
    self.threadCount = Int(threadCount)
  }

  /**
   Loads the labels from the labels file and stores it in an instance variable
   */
  func loadLabels(fromFileName fileName: String, fileExtension: String) {

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

  /**
 This assigns color for a particular class.
 */
  private func colorForClass(withIndex index: Int) -> UIColor {

    // We have a set of colors and the depending upon a stride, it assigns variations to of the base colors to each object based on it's index.
    let baseColor = colors[index % colors.count]

    var colorToAssign = baseColor

    let percentage = CGFloat((colorStrideValue / 2 - index / colors.count) * colorStrideValue)

    if let modifiedColor = baseColor.getModified(byPercentage: percentage) {
      colorToAssign = modifiedColor
    }

    return colorToAssign
  }
}
