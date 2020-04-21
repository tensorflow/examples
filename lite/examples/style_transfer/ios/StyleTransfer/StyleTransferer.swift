// Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

class StyleTransferer {

  /// TensorFlow Lite `Interpreter` object for performing inference on a given model.
  private var predictInterpreter: Interpreter
  private var transferInterpreter: Interpreter

  /// Dedicated DispatchQueue for TF Lite operations.
  private let tfLiteQueue: DispatchQueue

  // MARK: - Initialization

  /// Create a new Style Transferer instance.
  static func newInstance(completion: @escaping ((Result<StyleTransferer>) -> Void)) {
    // Create a dispatch queue to ensure all operations on the Intepreter will run serially.
    let tfLiteQueue = DispatchQueue(label: "org.tensorflow.examples.lite.style_transfer")

    // Run initialization in background thread to avoid UI freeze.
    tfLiteQueue.async {
      // Construct the path to the model file.
      let transferModelPath = Bundle.main.path(
        forResource: Constants.transferModelFileName,
        ofType: Constants.modelFileExtension
      )
      let predictModelPath = Bundle.main.path(
        forResource: Constants.predictModelFileName,
        ofType: Constants.modelFileExtension
      )
      guard let transferModel = transferModelPath, let predictModel = predictModelPath
      else {
        var errorMessage: [String] = []
        if transferModelPath == nil {
          errorMessage.append("\(Constants.transferModelFileName).\(Constants.modelFileExtension)")
        }
        if predictModelPath == nil {
          errorMessage.append("\(Constants.predictModelFileName).\(Constants.modelFileExtension)")
        }
        completion(.error(InitializationError.invalidModel(errorMessage.joined(separator: ", "))))
        return
      }

      // Specify the options for the TF Lite `Interpreter`.
      var options: Interpreter.Options?
      var delegates: [Delegate]?
#if targetEnvironment(simulator)
      // Use CPU for inference as MetalDelegate does not support iOS simulator.
      options = Interpreter.Options()
      options?.threadCount = 8
#else
      // Use GPU on real device for inference as this model is fully supported.
      delegates = [MetalDelegate()]
#endif

      do {
        // Create the `Interpreter`s.
        let predictInterpreter = try Interpreter(
          modelPath: predictModel,
          options: options,
          delegates: delegates
        )
        let transferInterpreter = try Interpreter(
          modelPath: transferModel,
          options: options,
          delegates: delegates
        )

        // Allocate memory for the model's input `Tensor`s.
        try predictInterpreter.allocateTensors()
        try transferInterpreter.allocateTensors()

        // Create an StyleTransferer instance and return.
        let styleTransferer = StyleTransferer(
          tfLiteQueue: tfLiteQueue,
          predictInterpreter: predictInterpreter,
          transferInterpreter: transferInterpreter
        )
        DispatchQueue.main.async {
          completion(.success(styleTransferer))
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

  /// Initialize Style Transferer instance.
  fileprivate init(
    tfLiteQueue: DispatchQueue,
    predictInterpreter: Interpreter,
    transferInterpreter: Interpreter
  ) {
    // Store TF Lite intepreter
    self.predictInterpreter = predictInterpreter
    self.transferInterpreter = transferInterpreter

    // Store the dedicated DispatchQueue for TFLite.
    self.tfLiteQueue = tfLiteQueue
  }

  // MARK: - Style Transfer

  /// Run style transfer on a given image.
  /// - Parameter styleImage: the image to use as a style reference.
  /// - Parameter image: the target image.
  /// - Parameter completion: the callback to receive the style transfer result.
  func runStyleTransfer(
    style styleImage: UIImage, image: UIImage, completion: @escaping ((Result<StyleTransferResult>) -> Void)
  ) {
    tfLiteQueue.async {
      let outputTensor: Tensor
      let inputSizeInPixels: CGSize
      let startTime: Date = Date()
      var preprocessingTime: TimeInterval = 0
      var stylePredictTime: TimeInterval = 0
      var styleTransferTime: TimeInterval = 0
      var postprocessingTime: TimeInterval = 0

      func timeSinceStart() -> TimeInterval {
        return abs(startTime.timeIntervalSinceNow)
      }

      do {
        // Preprocess style image.
        guard
          let styleRGBData = styleImage.scaledData(
            with: Constants.styleImageSize,
            isQuantized: false
          )
        else {
          DispatchQueue.main.async {
            completion(.error(StyleTransferError.invalidImage))
          }
          print("Failed to convert the style image buffer to RGB data.")
          return
        }

        guard
          let inputRGBData = image.scaledData(
            with: Constants.inputImageSize,
            isQuantized: false
          )
        else {
          DispatchQueue.main.async {
            completion(.error(StyleTransferError.invalidImage))
          }
          print("Failed to convert the input image buffer to RGB data.")
          return
        }

        preprocessingTime = timeSinceStart()

        // Copy the RGB data to the input `Tensor`.
        try self.predictInterpreter.copy(styleRGBData, toInputAt: 0)

        // Run inference by invoking the `Interpreter`.
        try self.predictInterpreter.invoke()

        // Get the output `Tensor` to process the inference results.
        let predictResultTensor = try self.predictInterpreter.output(at: 0)

        // Grab bottleneck data from output tensor.
        let bottleneck = predictResultTensor.data

        stylePredictTime = timeSinceStart() - preprocessingTime

        // Copy the RGB and bottleneck data to the input `Tensor`.
        try self.transferInterpreter.copy(inputRGBData, toInputAt: 0)
        try self.transferInterpreter.copy(bottleneck, toInputAt: 1)

        // Run inference by invoking the `Interpreter`.
        try self.transferInterpreter.invoke()

        // Get the result tensor
        outputTensor = try self.transferInterpreter.output(at: 0)

        styleTransferTime = timeSinceStart() - stylePredictTime - preprocessingTime

      } catch let error {
        print("Failed to invoke the interpreter with error: \(error.localizedDescription)")
        DispatchQueue.main.async {
          completion(.error(StyleTransferError.internalError(error)))
        }
        return
      }

      // Construct image from output tensor data
      guard let cgImage = self.postprocessImageData(data: outputTensor.data) else {
        DispatchQueue.main.async {
          completion(.error(StyleTransferError.resultVisualizationError))
        }
        return
      }

      let outputImage = UIImage(cgImage: cgImage)

      postprocessingTime =
          timeSinceStart() - stylePredictTime - styleTransferTime - preprocessingTime

      // Return the result.
      DispatchQueue.main.async {
        completion(
          .success(
            StyleTransferResult(
              resultImage: outputImage,
              preprocessingTime: preprocessingTime,
              stylePredictTime: stylePredictTime,
              styleTransferTime: styleTransferTime,
              postprocessingTime: postprocessingTime
            )
          )
        )
      }
    }
  }

    // MARK: - Utils

  /// Turns TF model's float32 array output into one supported by `CGImage`. This method
  /// assumes the provided data is the same format as the data returned from the output
  /// tensor in `runStyleTransfer`, so it should not be used for general image processing.
  /// - Parameter data: The image data to turn into a `CGImage`. This data must be a buffer of
  ///   `Float32` values between 0 and 1 in RGB format.
  /// - Parameter size: The expected size of the output image.
  private func postprocessImageData(data: Data,
                                    size: CGSize = Constants.inputImageSize) -> CGImage? {
    var width = Int(size.width)
    var height = Int(size.height)

    let floats = data.toArray(type: Float32.self)

    let bufferCapacity = width * height * 4
    let unsafePointer = UnsafeMutablePointer<UInt8>.allocate(capacity: bufferCapacity)
    let unsafeBuffer = UnsafeMutableBufferPointer<UInt8>(start: unsafePointer,
                                                         count: bufferCapacity)
    defer {
      unsafePointer.deallocate()
    }

    for x in 0 ..< width {
      for y in 0 ..< height {
        let floatIndex = (y * width + x) * 3
        let index = (y * width + x) * 4
        let red = UInt8(floats[floatIndex] * 255)
        let green = UInt8(floats[floatIndex + 1] * 255)
        let blue = UInt8(floats[floatIndex + 2] * 255)

        unsafeBuffer[index] = red
        unsafeBuffer[index + 1] = green
        unsafeBuffer[index + 2] = blue
        unsafeBuffer[index + 3] = 0
      }
    }

    let outData = Data(buffer: unsafeBuffer)

    // Construct image from output tensor data
    let alphaInfo = CGImageAlphaInfo.noneSkipLast
    let bitmapInfo = CGBitmapInfo(rawValue: alphaInfo.rawValue)
        .union(.byteOrder32Big)
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    guard
      let imageDataProvider = CGDataProvider(data: outData as CFData),
      let cgImage = CGImage(
        width: width,
        height: height,
        bitsPerComponent: 8,
        bitsPerPixel: 32,
        bytesPerRow: MemoryLayout<UInt8>.size * 4 * Int(Constants.inputImageSize.width),
        space: colorSpace,
        bitmapInfo: bitmapInfo,
        provider: imageDataProvider,
        decode: nil,
        shouldInterpolate: false,
        intent: .defaultIntent
      )
      else {
        return nil
    }
    return cgImage
  }

}

// MARK: - Types

/// Representation of the style transfer result.
struct StyleTransferResult {

  /// The resulting image from the style transfer.
  let resultImage: UIImage

  /// Time required to resize the input and style images and convert the image
  /// data to a format the model can accept.
  let preprocessingTime: TimeInterval

  /// The style prediction model run time.
  let stylePredictTime: TimeInterval

  /// The style transfer model run time.
  let styleTransferTime: TimeInterval

  /// Time required to convert the model output data to a `CGImage`.
  let postprocessingTime: TimeInterval

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

/// Define errors that could happen when running style transfer
enum StyleTransferError: Error {
  // Invalid input image
  case invalidImage

  // TF Lite Internal Error when initializing
  case internalError(Error)

  // Invalid input image
  case resultVisualizationError
}

// MARK: - Constants
private enum Constants {

  /// The TF Lite style prediction model file
  static let predictModelFileName = "style_predict_quantized_256"

  /// The TF Lite style transfer model file
  static let transferModelFileName = "style_transfer_quantized_384"

  static let modelFileExtension = "tflite"

  static let styleImageSize = CGSize(width: 256, height: 256)

  static let inputImageSize = CGSize(width: 384, height: 384)

}
