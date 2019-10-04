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

import TensorFlowLite
import UIKit

class ImageSegmentator {

  /// TensorFlow Lite `Interpreter` object for performing inference on a given model.
  private var interpreter: Interpreter

  /// TF Lite Model's input and output shapes.
  private let batchSize: Int
  private let inputImageWidth: Int
  private let inputImageHeight: Int
  private let inputPixelSize: Int
  private let outputImageWidth: Int
  private let outputImageHeight: Int
  private let outputClassCount: Int

  /// Label list contains name of all classes the model can regconize.
  private let labelList: [String]

  // MARK: - Initialization

  /// Load label list from file.
  private static func loadLabelList() -> [String]? {
    guard
      let labelListPath = Bundle.main.path(
        forResource: Constants.labelsFileName,
        ofType: Constants.labelsFileExtension
      )
    else {
      return nil
    }

    // Parse label list file as JSON.
    do {
      let data = try Data(contentsOf: URL(fileURLWithPath: labelListPath), options: .mappedIfSafe)
      let jsonResult = try JSONSerialization.jsonObject(with: data, options: .mutableLeaves)
      if let labelList = jsonResult as? [String] { return labelList } else { return nil }
    } catch {
      print("Error parsing label list file as JSON.")
      return nil
    }
  }

  /// Create a new Image Segmentator instance.
  static func newInstance(completion: @escaping ((Result<ImageSegmentator>) -> Void)) {
    // Run initialization in background thread to avoid UI freeze.
    DispatchQueue.global(qos: .background).async {
      // Construct the path to the model file.
      guard
        let modelPath = Bundle.main.path(
          forResource: Constants.modelFileName,
          ofType: Constants.modelFileExtension
        )
      else {
        print(
          "Failed to load the model file with name: "
            + "\(Constants.modelFileName).\(Constants.modelFileExtension)")
        DispatchQueue.main.async {
          completion(
            .error(
              InitializationError.invalidModel(
                "\(Constants.modelFileName).\(Constants.modelFileExtension)"
              )))
        }
        return
      }

      // Construct the path to the label list file.
      guard let labelList = loadLabelList() else {
        print(
          "Failed to load the label list file with name: "
            + "\(Constants.labelsFileName).\(Constants.labelsFileExtension)"
        )
        DispatchQueue.main.async {
          completion(
            .error(
              InitializationError.invalidLabelList(
                "\(Constants.labelsFileName).\(Constants.labelsFileExtension)"
              )))
        }
        return
      }

      // Specify the options for the TF Lite `Interpreter`.
      var options = InterpreterOptions()
      options.threadCount = 2

      do {
        // Create the `Interpreter`.
        let interpreter = try Interpreter(modelPath: modelPath, options: options)

        // Allocate memory for the model's input `Tensor`s.
        try interpreter.allocateTensors()

        // Read TF Lite model input and output shapes.
        let inputShape = try interpreter.input(at: 0).shape
        let outputShape = try interpreter.output(at: 0).shape

        // Create an ImageSegmentator instance and return.
        let segmentator = ImageSegmentator(
          interpreter: interpreter,
          inputShape: inputShape,
          outputShape: outputShape,
          labelList: labelList
        )
        DispatchQueue.main.async {
          completion(.success(segmentator))
        }
      } catch let error {
        print("Failed to create the interpreter with error: \(error.localizedDescription)")
        DispatchQueue.main.async {
          completion(.error(InitializationError.internalError(error)))
        }
        return
      }
    }
  }

  /// Initialize Image Segmentator instance.
  fileprivate init(
    interpreter: Interpreter,
    inputShape: TensorShape,
    outputShape: TensorShape,
    labelList: [String]
  ) {
    // Store TF Lite intepreter
    self.interpreter = interpreter

    // Read input shape from model.
    self.batchSize = inputShape.dimensions[0]
    self.inputImageWidth = inputShape.dimensions[1]
    self.inputImageHeight = inputShape.dimensions[2]
    self.inputPixelSize = inputShape.dimensions[3]

    // Read output shape from model
    self.outputImageWidth = outputShape.dimensions[1]
    self.outputImageHeight = outputShape.dimensions[2]
    self.outputClassCount = outputShape.dimensions[3]

    // Store label list
    self.labelList = labelList
  }

  // MARK: - Image Segmentation

  /// Run segmentation on a given image.
  /// - Parameter image: the target image.
  /// - Parameter completion: the callback to receive segmentation result.
  func runSegmentation(
    _ image: UIImage, completion: @escaping ((Result<SegmentationResult>) -> Void)
  ) {
    DispatchQueue.global(qos: .background).async {
      let outputTensor: Tensor
      var startTime: Date = Date()
      var preprocessingTime: TimeInterval = 0
      var inferenceTime: TimeInterval = 0
      var postprocessingTime: TimeInterval = 0
      var visualizationTime: TimeInterval = 0

      do {
        // Preprocessing: Resize the input UIImage to match with TF Lite model input shape.
        guard
          let rgbData = image.scaledData(
            with: CGSize(width: self.inputImageWidth, height: self.inputImageHeight),
            byteCount: self.inputImageWidth * self.inputImageHeight * self.inputPixelSize
              * self.batchSize,
            isQuantized: false
          )
        else {
          DispatchQueue.main.async {
            completion(.error(SegmentationError.invalidImage))
          }
          print("Failed to convert the image buffer to RGB data.")
          return
        }

        // Calculate preprocessing time.
        var now = Date()
        preprocessingTime = now.timeIntervalSince(startTime)
        startTime = Date()

        // Allocate memory for the model's input `Tensor`s.
        try self.interpreter.allocateTensors()

        // Copy the RGB data to the input `Tensor`.
        try self.interpreter.copy(rgbData, toInputAt: 0)

        // Run inference by invoking the `Interpreter`.
        try self.interpreter.invoke()

        // Get the output `Tensor` to process the inference results.
        outputTensor = try self.interpreter.output(at: 0)

        // Calculate inference time.
        now = Date()
        inferenceTime = now.timeIntervalSince(startTime)
        startTime = Date()
      } catch let error {
        print("Failed to invoke the interpreter with error: \(error.localizedDescription)")
        DispatchQueue.main.async {
          completion(.error(SegmentationError.internalError(error)))
        }
        return
      }

      // Postprocessing: Find the class with highest confidence for each pixel.
      let parsedOutput = self.parseOutputTensor(outputTensor: outputTensor)

      // Calculate postprocessing time.
      // Note: You may find postprocessing very slow if you run the sample app with Debug build.
      // You will see significant speed up if you rerun using Release build, or change
      // Optimization Level in the project's Build Settings to the same value with Release build.
      var now = Date()
      postprocessingTime = now.timeIntervalSince(startTime)
      startTime = Date()

      // Visualize result into images.
      guard
        let resultImage = ImageSegmentator.imageFromSRGBColorArray(
          pixels: parsedOutput.segmentationImagePixels,
          width: self.inputImageWidth,
          height: self.inputImageHeight
        ),
        let overlayImage = image.overlayWithImage(image: resultImage, alpha: 0.5)
      else {
        print("Failed to visualize segmentation result.")
        DispatchQueue.main.async {
          completion(.error(SegmentationError.resultVisualizationError))
        }
        return
      }

      // Construct a dictionary of classes found in the image and each class's color used in
      // visualization.
      let colorLegend = self.classListToColorLegend(classList: parsedOutput.classList)

      // Calculate visualization time.
      now = Date()
      visualizationTime = now.timeIntervalSince(startTime)

      // Create a representative object that contains the segmentation result.
      let result = SegmentationResult(
        array: parsedOutput.segmentationMap,
        resultImage: resultImage,
        overlayImage: overlayImage,
        preprocessingTime: preprocessingTime,
        inferenceTime: inferenceTime,
        postProcessingTime: postprocessingTime,
        visualizationTime: visualizationTime,
        colorLegend: colorLegend
      )

      // Return the segmentation result.
      DispatchQueue.main.async {
        completion(.success(result))
      }
    }
  }

  /// Post-processing: Convert TensorFlow Lite output tensor to segmentation map and its color
  /// representation.
  private func parseOutputTensor(outputTensor: Tensor)
    -> (segmentationMap: [[Int]], segmentationImagePixels: [UInt32], classList: Set<Int>)
  {
    // Initialize the varibles to store postprocessing result.
    var segmentationMap = [[Int]](
      repeating: [Int](repeating: 0, count: self.outputImageHeight),
      count: self.outputImageWidth
    )
    var segmentationImagePixels = [UInt32](
      repeating: 0, count: self.outputImageHeight * self.outputImageWidth)
    var classList: Set<Int> = []

    // Convert TF Lite model output to a native Float32 array for parsing.
    let logits = outputTensor.data.toArray(type: Float32.self)

    var valMax: Float32 = 0.0
    var val: Float32 = 0.0
    var indexMax = 0

    // Looping through the output array
    for x in 0..<self.outputImageWidth {
      for y in 0..<self.outputImageHeight {
        // For each pixel, find the class that have the highest probability.
        valMax = logits[self.coordinateToIndex(x: x, y: y, z: 0)]
        indexMax = 0
        for z in 1..<self.outputClassCount {
          val = logits[self.coordinateToIndex(x: x, y: y, z: z)]
          if logits[self.coordinateToIndex(x: x, y: y, z: z)] > valMax {
            indexMax = z
            valMax = val
          }
        }

        // Store the most likely class to the output.
        segmentationMap[x][y] = indexMax
        classList.insert(indexMax)

        // Lookup the color legend for the class.
        // Using modulo to reuse colors on segmentation model with large number of classes.
        let legendColor = Constants.legendColorList[indexMax % Constants.legendColorList.count]
        segmentationImagePixels[x * self.outputImageHeight + y] = legendColor
      }
    }

    return (segmentationMap, segmentationImagePixels, classList)
  }

  // MARK: - Utils

  /// Construct an UIImage from a list of sRGB pixels.
  private static func imageFromSRGBColorArray(pixels: [UInt32], width: Int, height: Int) -> UIImage?
  {
    guard width > 0 && height > 0 else { return nil }
    guard pixels.count == width * height else { return nil }

    // Make a mutable copy
    var data = pixels

    // Convert array of pixels to a CGImage instance.
    let cgImage = data.withUnsafeMutableBytes { (ptr) -> CGImage in
      let ctx = CGContext(
        data: ptr.baseAddress,
        width: width,
        height: height,
        bitsPerComponent: 8,
        bytesPerRow: MemoryLayout<UInt32>.size * width,
        space: CGColorSpace(name: CGColorSpace.sRGB)!,
        bitmapInfo: CGBitmapInfo.byteOrder32Little.rawValue
          + CGImageAlphaInfo.premultipliedFirst.rawValue
      )!
      return ctx.makeImage()!
    }

    // Convert the CGImage instance to an UIImage instance.
    return UIImage(cgImage: cgImage)
  }

  /// Convert 3-dimension index (image_width x image_height x class_count) to 1-dimension index
  private func coordinateToIndex(x: Int, y: Int, z: Int) -> Int {
    return x * outputImageHeight * outputClassCount + y * outputClassCount + z
  }

  /// Look up the colors used to visualize the classes found in the image.
  private func classListToColorLegend(classList: Set<Int>) -> [String: UIColor] {
    var colorLegend: [String: UIColor] = [:]
    let sortedClassIndexList = classList.sorted()
    sortedClassIndexList.forEach { classIndex in
      // Look up the color legend for the class.
      // Using modulo to reuse colors on segmentation model with large number of classes.
      let color = Constants.legendColorList[classIndex % Constants.legendColorList.count]

      // Convert the color from sRGB UInt32 representation to UIColor.
      let a = CGFloat((color & 0xFF00_0000) >> 24) / 255.0
      let r = CGFloat((color & 0x00FF_0000) >> 16) / 255.0
      let g = CGFloat((color & 0x0000_FF00) >> 8) / 255.0
      let b = CGFloat(color & 0x0000_00FF) / 255.0
      colorLegend[labelList[classIndex]] = UIColor(red: r, green: g, blue: b, alpha: a)
    }
    return colorLegend
  }

}

// MARK: - Types

/// Callback type for image segmentation request.
typealias ImageSegmentationCompletion = (SegmentationResult?, Error?) -> Void

/// Representation of the image segmentation result.
struct SegmentationResult {
  /// Segmentation result as an array. Each value represents the most likely class the pixel
  /// belongs to.
  let array: [[Int]]

  /// Visualization of the segmentation result.
  let resultImage: UIImage

  /// Overlay the segmentation result on input image.
  let overlayImage: UIImage

  /// Processing time.
  let preprocessingTime: TimeInterval
  let inferenceTime: TimeInterval
  let postProcessingTime: TimeInterval
  let visualizationTime: TimeInterval

  /// Dictionary of classes found in the image, and the color used to represent the class in
  /// segmentation result visualization.
  let colorLegend: [String: UIColor]
}

/// Convenient enum to return result with a callback
enum Result<T> {
  case success(T)
  case error(Error)
}

/// Define errors that could happen in the initialization of this class
enum InitializationError: Error {
  // Invalid TF Lite model
  case invalidModel(String)

  // Invalid label list
  case invalidLabelList(String)

  // TF Lite Internal Error when initializing
  case internalError(Error)
}

/// Define errors that could happen in when doing image segmentation
enum SegmentationError: Error {
  // Invalid input image
  case invalidImage

  // TF Lite Internal Error when initializing
  case internalError(Error)

  // Invalid input image
  case resultVisualizationError
}

// MARK: - Constants
private enum Constants {
  /// Label list that the segmentation model detects.
  static let labelsFileName = "deeplabv3_labels"
  static let labelsFileExtension = "json"

  /// The TF Lite segmentation model file
  static let modelFileName = "deeplabv3_257_mv_gpu"
  static let modelFileExtension = "tflite"

  /// List of colors to visualize segmentation result.
  static let legendColorList: [UInt32] = [
    0xFFFF_B300, // Vivid Yellow
    0xFF80_3E75, // Strong Purple
    0xFFFF_6800, // Vivid Orange
    0xFFA6_BDD7, // Very Light Blue
    0xFFC1_0020, // Vivid Red
    0xFFCE_A262, // Grayish Yellow
    0xFF81_7066, // Medium Gray
    0xFF00_7D34, // Vivid Green
    0xFFF6_768E, // Strong Purplish Pink
    0xFF00_538A, // Strong Blue
    0xFFFF_7A5C, // Strong Yellowish Pink
    0xFF53_377A, // Strong Violet
    0xFFFF_8E00, // Vivid Orange Yellow
    0xFFB3_2851, // Strong Purplish Red
    0xFFF4_C800, // Vivid Greenish Yellow
    0xFF7F_180D, // Strong Reddish Brown
    0xFF93_AA00, // Vivid Yellowish Green
    0xFF59_3315, // Deep Yellowish Brown
    0xFFF1_3A13, // Vivid Reddish Orange
    0xFF23_2C16, // Dark Olive Green
    0xFF00_A1C2, // Vivid Blue
  ]
}
