//
//  StylePredictorModelDataHandler.swift
//  ImageClassification
//
//  Created by Ivan Cheung on 3/28/20.
//  Copyright © 2020 Y Media Labs. All rights reserved.
//

import UIKit
import CoreImage
import TensorFlowLite

enum Style: String, CaseIterable {
  case style0, style1, style2, style3, style4, style5, style6, style7
  case style8, style9, style10, style11, style12, style13, style14
  case style15, style16, style17, style18, style19, style20, style21
  case style22, style23, style24, style25
}

/// Information about the MobileNet model.
enum StylePredictorModel {
  static let modelInfo: FileInfo = (name: "style_predict_f16_256", extension: "tflite")
  static let modelInfoQuantized: FileInfo = (name: "style_predict_quantized_256", extension: "tflite")
}

typealias StyleBottleneck = [Float]

/// This class handles all data preprocessing and makes calls to run inference on a given frame
/// by invoking the `Interpreter`. It then formats the inferences obtained and returns the top N
/// results for a successful inference.
class StylePredictorModelDataHandler: ModelDataHandling {
  typealias Inference = StyleBottleneck
    
  // MARK: - Internal Properties
  /// The current thread count used by the TensorFlow Lite Interpreter.
  let threadCount: Int

  let resultCount = 3

  // MARK: - Model Parameters

  let batchSize = 1
  let inputChannels = 3
  let inputWidth = 256
  let inputHeight = 256

//  let bottleneckSize = 100

  // MARK: - Private Properties

  /// TensorFlow Lite `Interpreter` object for performing inference on a given model.
  private var interpreter: Interpreter

  /// Information about the alpha component in RGBA data.
  private let alphaComponent = (baseOffset: 4, moduloRemainder: 3)
  
  private let modelFileInfo = StylePredictorModel.modelInfo
  
  // MARK: - Initialization

  /// A failable initializer for `ModelDataHandler`. A new instance is created if the model and
  /// labels files are successfully loaded from the app's main bundle. Default `threadCount` is 1.
  init?(threadCount: Int = 1) {
    let modelFilename = modelFileInfo.name

    // Construct the path to the model file.
    guard let modelPath = Bundle.main.path(
      forResource: modelFilename,
      ofType: modelFileInfo.extension
    ) else {
      print("Failed to load the model file with name: \(modelFilename).")
      return nil
    }

    // Specify the options for the `Interpreter`.
    self.threadCount = threadCount
    var options = InterpreterOptions()
    options.threadCount = threadCount
    do {
      // Create the `Interpreter`.
      interpreter = try Interpreter(modelPath: modelPath, options: options)
            
      // Allocate memory for the model's input `Tensor`s.
      try interpreter.allocateTensors()
      
      for tensorIndex in 0..<interpreter.inputTensorCount {
        print("\(tensorIndex): \(try interpreter.input(at: tensorIndex).shape)")
      }
    } catch let error {
      print("Failed to create the interpreter with error: \(error.localizedDescription)")
      return nil
    }
  }

  // MARK: - Internal Methods

  /// Performs image preprocessing, invokes the `Interpreter`, and processes the inference results.
  func runModel(input style: Style) -> Result<StyleBottleneck>? {
    guard let image = UIImage(named: style.rawValue),
          let pixelBuffer = CVPixelBuffer.buffer(from: image) else { return nil }
    
    let sourcePixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer)
    assert(sourcePixelFormat == kCVPixelFormatType_32ARGB ||
             sourcePixelFormat == kCVPixelFormatType_32BGRA ||
               sourcePixelFormat == kCVPixelFormatType_32RGBA)


    let imageChannels = 4
    assert(imageChannels >= inputChannels)

    // Crops the image to the biggest square in the center and scales it down to model dimensions.
    let scaledSize = CGSize(width: inputWidth, height: inputHeight)
    guard let thumbnailPixelBuffer = pixelBuffer.centerThumbnail(ofSize: scaledSize) else {
      return nil
    }

    let interval: TimeInterval
    let outputTensor: Tensor
    do {
      let inputTensor = try interpreter.input(at: 0)

      // Remove the alpha component from the image buffer to get the RGB data.
      guard let rgbData = ImageHelper.rgbDataFromBuffer(
        thumbnailPixelBuffer,
        byteCount: batchSize * inputWidth * inputHeight * inputChannels,
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

    // Handle quantization
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
    // Return the inference time and inference results.
//    let logInfo = StyleBottleneck.LogInfo(preProcessTime: <#T##Int#>,
//                                                  stylePredictTime: <#T##Int#>,
//                                                  styleTransferTime: <#T##Int#>,
//                                                  postProcessTime: <#T##Int#>,
//                                                  totalExecutionTime: <#T##Int#>,
//                                                  executionLog: <#T##String#>,
//                                                  errorMessage: <#T##String#>)
    
    return Result<StyleBottleneck>(elapsedTimeInMs: interval, inference: results)
  }
}
