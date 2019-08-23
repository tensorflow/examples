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

import Accelerate
import CoreImage
import TensorFlowLite
import UIKit

// MARK: Structures That hold results
/**
 Stores inference results for a particular frame that was successfully run through the
 TensorFlow Lite Interpreter.
 */
struct Result{
  let inferenceTime: Double
  let inferences: [Inference]
}

/**
 Stores one formatted inference.
 */
struct Inference {
  let confidence: Float
  let label: String
}

/// Information about a model file or labels file.
typealias FileInfo = (name: String, extension: String)

// Information about the model to be loaded.
enum Model {
  static let modelInfo: FileInfo = (name: "model", extension: "tflite")
  static let labelsInfo: FileInfo = (name: "labels", extension: "txt")
}

/**
 This class handles all data preprocessing and makes calls to run inference on
 a given frame through the TensorFlow Lite Interpreter. It then formats the
 inferences obtained and returns the top N results for a successful inference.
 */
class ModelDataHandler {

  // MARK: Paremeters on which model was trained
  let batchSize = 1
  let wantedInputChannels = 3
  let wantedInputWidth = 224
  let wantedInputHeight = 224
  let stdDeviation: Float = 127.0
  let mean: Float = 1.0

  // MARK: Constants
  let threadCountLimit: Int32 = 10

  // MARK: Instance Variables
  /// The current thread count used by the TensorFlow Lite Interpreter.
  let threadCount: Int

  var labels: [String] = []
  private let resultCount = 1
  private let threshold = 0.5

  /// TensorFlow Lite `Interpreter` object for performing inference on a given model.
  private var interpreter: Interpreter

  private let bgraPixel = (channels: 4, alphaComponent: 3, lastBgrComponent: 2)
  private let rgbPixelChannels = 3
  private let colorStrideValue = 10

  /// Information about the alpha component in RGBA data.
  private let alphaComponent = (baseOffset: 4, moduloRemainder: 3)

  // MARK: Initializer
  /**
   This is a failable initializer for ModelDataHandler. It successfully initializes an object of the class if the model file and labels file is found, labels can be loaded and the interpreter of TensorflowLite can be initialized successfully.
   */
  init?(modelFileInfo: FileInfo, labelsFileInfo: FileInfo, threadCount: Int = 1) {
    // Construct the path to the model file.
    guard let modelPath = Bundle.main.path(
      forResource: modelFileInfo.name,
      ofType: modelFileInfo.extension
    ) else {
      print("Failed to load the model file with name: \(modelFileInfo.name).")
      return nil
    }

    // Specify the options for the `Interpreter`.
    self.threadCount = threadCount
    var options = Interpreter.Options()
    options.threadCount = threadCount
    do {
      // Create the `Interpreter`.
      interpreter = try Interpreter(modelPath: modelPath, options: options)
      // Allocate memory for the model's input `Tensor`s.
      try interpreter.allocateTensors()
    } catch let error {
      print("Failed to create the interpreter with error: \(error.localizedDescription)")
      return nil
    }

    // Opens and loads the classes listed in labels file
    loadLabels(fromFileName: Model.labelsInfo.name, fileExtension: Model.labelsInfo.extension)
  }

  // MARK: Methods for data preprocessing and post processing.
  /**
   Performs image preprocessing, calls the TensorFlow Lite Interpreter methods
   to feed the pixel buffer into the input tensor and  run inference
   on the pixel buffer.
   */
  func runModel(onFrame pixelBuffer: CVPixelBuffer) -> Result? {

    let sourcePixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer)
    assert(sourcePixelFormat == kCVPixelFormatType_32ARGB ||
      sourcePixelFormat == kCVPixelFormatType_32BGRA || sourcePixelFormat == kCVPixelFormatType_32RGBA)


    let imageChannels = 4
    assert(imageChannels >= wantedInputChannels)

    // Crops the image to the biggest square in the center and scales it down to model dimensions.
    guard let thumbnailPixelBuffer = centerThumbnail(from: pixelBuffer, size: CGSize(width: wantedInputWidth, height: wantedInputHeight)) else {
      return nil
    }

    CVPixelBufferLockBaseAddress(thumbnailPixelBuffer, CVPixelBufferLockFlags(rawValue: 0))

    let interval: TimeInterval
    let outputTensor: Tensor
    do {
      let inputTensor = try interpreter.input(at: 0)

      // Remove the alpha component from the image buffer to get the RGB data.
      guard let rgbData = rgbDataFromBuffer(
        thumbnailPixelBuffer,
        byteCount: batchSize * wantedInputWidth * wantedInputHeight * wantedInputChannels,
        isModelQuantized: inputTensor.dataType == .uInt8
        ) else {
          print("Failed to convert the image buffer to RGB data.")
          return nil
      }

      // Copy the RGB data to the input `Tensor`.
      try interpreter.copy(rgbData, toInputAt: 0)

      // Run inference by invoking the `Interpreter`.
      let startDate = Date()
      try interpreter.invoke()
      interval = Date().timeIntervalSince(startDate) * 1000

      // Get the output `Tensor` to process the inference results.
      outputTensor = try interpreter.output(at: 0)
    } catch let error {
      print("Failed to invoke the interpreter with error: \(error.localizedDescription)")
      return nil
    }

    let results: [Float]
    switch outputTensor.dataType {
    case .uInt8:
      guard let quantization = outputTensor.quantizationParameters else {
        print("No results returned because the quantization values for the output tensor are nil.")
        return nil
      }
      let quantizedResults = [UInt8](outputTensor.data)
      results = quantizedResults.map {
        quantization.scale * Float(Int($0) - quantization.zeroPoint)
      }
    case .float32:
      results = [Float32](unsafeData: outputTensor.data) ?? []
    default:
      print("Output tensor data type \(outputTensor.dataType) is unsupported for this example app.")
      return nil
    }

    // Process the results.
    let topNInferences = getTopN(results: results)

    // Return the inference time and inference results.
    return Result(inferenceTime: interval, inferences: topNInferences)
  }

  /// Returns the RGB data representation of the given image buffer with the specified `byteCount`.
  ///
  /// - Parameters
  ///   - buffer: The BGRA pixel buffer to convert to RGB data.
  ///   - byteCount: The expected byte count for the RGB data calculated using the values that the
  ///       model was trained on: `batchSize * imageWidth * imageHeight * componentsCount`.
  ///   - isModelQuantized: Whether the model is quantized (i.e. fixed point values rather than
  ///       floating point values).
  /// - Returns: The RGB data representation of the image buffer or `nil` if the buffer could not be
  ///     converted.
  private func rgbDataFromBuffer(
    _ buffer: CVPixelBuffer,
    byteCount: Int,
    isModelQuantized: Bool
  ) -> Data? {
    CVPixelBufferLockBaseAddress(buffer, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
    guard let mutableRawPointer = CVPixelBufferGetBaseAddress(buffer) else {
      return nil
    }
    assert(CVPixelBufferGetPixelFormatType(buffer) == kCVPixelFormatType_32BGRA)
    let count = CVPixelBufferGetDataSize(buffer)
    let bufferData = Data(bytesNoCopy: mutableRawPointer, count: count, deallocator: .none)
    var rgbBytes = [UInt8](repeating: 0, count: byteCount)
    var pixelIndex = 0
    for component in bufferData.enumerated() {
      let bgraComponent = component.offset % bgraPixel.channels;
      let isAlphaComponent = bgraComponent == bgraPixel.alphaComponent;
      guard !isAlphaComponent else {
        pixelIndex += 1
        continue
      }
      // Swizzle BGR -> RGB.
      let rgbIndex = pixelIndex * rgbPixelChannels + (bgraPixel.lastBgrComponent - bgraComponent)
      rgbBytes[rgbIndex] = component.element
    }
    if isModelQuantized { return Data(bytes: rgbBytes) }
    return Data(copyingBufferOf: rgbBytes.map { Float($0) / 255.0 })
  }

  /// Returns the top N inference results sorted in descending order.
  private func getTopN(results: [Float]) -> [Inference] {
    // Create a zipped array of tuples [(labelIndex: Int, confidence: Float)].
    let zippedResults = zip(labels.indices, results)

    // Sort the zipped results by confidence value in descending order.
    let sortedResults = zippedResults.sorted { $0.1 > $1.1 }.prefix(resultCount)

    // Return the `Inference` results.
    return sortedResults.map { result in Inference(confidence: result.1, label: labels[result.0]) }
  }

  /**
   Returns thumbnail by cropping pixel buffer to biggest square and scaling the cropped image to model dimensions.
   */
  private func centerThumbnail(from pixelBuffer: CVPixelBuffer, size: CGSize ) -> CVPixelBuffer? {

    let imageWidth = CVPixelBufferGetWidth(pixelBuffer)
    let imageHeight = CVPixelBufferGetHeight(pixelBuffer)
    let pixelBufferType = CVPixelBufferGetPixelFormatType(pixelBuffer)

    assert(pixelBufferType == kCVPixelFormatType_32BGRA)

    let inputImageRowBytes = CVPixelBufferGetBytesPerRow(pixelBuffer)
    let imageChannels = 4

    let thumbnailSize = min(imageWidth, imageHeight)
    CVPixelBufferLockBaseAddress(pixelBuffer, CVPixelBufferLockFlags(rawValue: 0))

    var originX = 0
    var originY = 0

    if imageWidth > imageHeight {
      originX = (imageWidth - imageHeight) / 2
    }
    else {
      originY = (imageHeight - imageWidth) / 2
    }

    // Finds the biggest square in the pixel buffer and advances rows based on it.
    guard let inputBaseAddress = CVPixelBufferGetBaseAddress(pixelBuffer)?.advanced(by: originY * inputImageRowBytes + originX * imageChannels) else {
      return nil
    }

    // Gets vImage Buffer from input image
    var inputVImageBuffer = vImage_Buffer(data: inputBaseAddress, height: UInt(thumbnailSize), width: UInt(thumbnailSize), rowBytes: inputImageRowBytes)

    let thumbnailRowBytes = Int(size.width) * imageChannels
    guard  let thumbnailBytes = malloc(Int(size.height) * thumbnailRowBytes) else {
      return nil
    }

    // Allocates a vImage buffer for thumbnail image.
    var thumbnailVImageBuffer = vImage_Buffer(data: thumbnailBytes, height: UInt(size.height), width: UInt(size.width), rowBytes: thumbnailRowBytes)

    // Performs the scale operation on input image buffer and stores it in thumbnail image buffer.
    let scaleError = vImageScale_ARGB8888(&inputVImageBuffer, &thumbnailVImageBuffer, nil, vImage_Flags(0))

    CVPixelBufferUnlockBaseAddress(pixelBuffer, CVPixelBufferLockFlags(rawValue: 0))

    guard scaleError == kvImageNoError else {
      return nil
    }

    let releaseCallBack: CVPixelBufferReleaseBytesCallback = {mutablePointer, pointer in

      if let pointer = pointer {
        free(UnsafeMutableRawPointer(mutating: pointer))
      }
    }

    var thumbnailPixelBuffer: CVPixelBuffer?

    // Converts the thumbnail vImage buffer to CVPixelBuffer
    let conversionStatus = CVPixelBufferCreateWithBytes(nil, Int(size.width), Int(size.height), pixelBufferType, thumbnailBytes, thumbnailRowBytes, releaseCallBack, nil, nil, &thumbnailPixelBuffer)

    guard conversionStatus == kCVReturnSuccess else {

      free(thumbnailBytes)
      return nil
    }

    return thumbnailPixelBuffer
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
      self.labels = contents.components(separatedBy: ",")
      self.labels.removeAll { (label) -> Bool in
        return label == ""
      }
    }
    catch {
      fatalError("Labels file named \(fileName).\(fileExtension) cannot be read. Please add a valid labels file and try again.")
    }

  }
}

// MARK: - Extensions

extension Data {
  /// Creates a new buffer by copying the buffer pointer of the given array.
  ///
  /// - Warning: The given array's element type `T` must be trivial in that it can be copied bit
  ///     for bit with no indirection or reference-counting operations; otherwise, reinterpreting
  ///     data from the resulting buffer has undefined behavior.
  /// - Parameter array: An array with elements of type `T`.
  init<T>(copyingBufferOf array: [T]) {
    self = array.withUnsafeBufferPointer(Data.init)
  }
}

extension Array {
  /// Creates a new array from the bytes of the given unsafe data.
  ///
  /// - Warning: The array's `Element` type must be trivial in that it can be copied bit for bit
  ///     with no indirection or reference-counting operations; otherwise, copying the raw bytes in
  ///     the `unsafeData`'s buffer to a new array returns an unsafe copy.
  /// - Note: Returns `nil` if `unsafeData.count` is not a multiple of
  ///     `MemoryLayout<Element>.stride`.
  /// - Parameter unsafeData: The data containing the bytes to turn into an array.
  init?(unsafeData: Data) {
    guard unsafeData.count % MemoryLayout<Element>.stride == 0 else { return nil }
    #if swift(>=5.0)
    self = unsafeData.withUnsafeBytes { .init($0.bindMemory(to: Element.self)) }
    #else
    self = unsafeData.withUnsafeBytes {
      .init(UnsafeBufferPointer<Element>(
        start: $0,
        count: unsafeData.count / MemoryLayout<Element>.stride
      ))
    }
    #endif  // swift(>=5.0)
  }
}
