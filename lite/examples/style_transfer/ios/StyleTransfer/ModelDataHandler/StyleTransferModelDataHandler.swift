//
//  StyleTransferModelDataHandler.swift
//  StyleTransfer
//
//  Created by Ivan Cheung on 3/28/20.
//  Copyright © 2020 Ivan Cheung. All rights reserved.
//

import CoreImage
import TensorFlowLite
import Accelerate

/// Information about the MobileNet model.
enum StyleTransferModel {
  static let modelInfo: FileInfo = (name: "style_transfer_f16_384", extension: "tflite")
  static let modelInfoQuantized: FileInfo = (name: "style_transfer_quantized_384", extension: "tflite")
}

struct StyleTransferInput {
  let styleBottleneck: [Float]
  let pixelBuffer: CVPixelBuffer
}
typealias StyleTransferOutput = UIImage

/// This class handles all data preprocessing and makes calls to perform style transfer on a given frame
/// by invoking the `Interpreter`. 
class StyleTransferModelDataHandler: ModelDataHandling {
  typealias Inference = StyleTransferOutput
  
  // MARK: - Internal Properties
  /// The current thread count used by the TensorFlow Lite Interpreter.
  let threadCount: Int
  
  let resultCount = 3
  
  // MARK: - Model Parameters
  
  let batchSize = 1
  let inputChannels = 3
  let contentImageSize = 384
  
  // MARK: - Private Properties
  
  /// TensorFlow Lite `Interpreter` object for performing inference on a given model.
  private var interpreter: Interpreter
  
  /// Information about the alpha component in RGBA data.
  private let alphaComponent = (baseOffset: 4, moduloRemainder: 3)
  
  private let modelFileInfo = StyleTransferModel.modelInfo
  
  // Cache
  
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
  func runModel(input: StyleTransferInput) -> Result<StyleTransferOutput>? {
    let pixelBuffer = input.pixelBuffer
    let sourcePixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer)
    assert(sourcePixelFormat == kCVPixelFormatType_32ARGB ||
      sourcePixelFormat == kCVPixelFormatType_32BGRA ||
      sourcePixelFormat == kCVPixelFormatType_32RGBA)
    
    
    let imageChannels = 4
    assert(imageChannels >= inputChannels)
    
    // Crops the image to the biggest square in the center and scales it down to model dimensions.
    let scaledSize = CGSize(width: contentImageSize, height: contentImageSize)
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
        byteCount: batchSize * contentImageSize * contentImageSize * inputChannels,
        isModelQuantized: inputTensor.dataType == .uInt8
        ) else {
          print("Failed to convert the image buffer to RGB data.")
          return nil
      }
      
      // Copy the RGB data to the input `Tensor`.
      try interpreter.copy(rgbData, toInputAt: 0)
      
      // Copy the 'style bottleneck' to the input `Tensor`
      try interpreter.copy(Data(copyingBufferOf: input.styleBottleneck), toInputAt: 1)
      
      
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
    let rgbDataAsFloats: [Float]
    switch outputTensor.dataType {
    case .uInt8:
      guard let quantization = outputTensor.quantizationParameters else {
        print("No results returned because the quantization values for the output tensor are nil.")
        return nil
      }
      let quantizedResults = [UInt8](outputTensor.data)
      rgbDataAsFloats = quantizedResults.map {
        quantization.scale * Float(Int($0) - quantization.zeroPoint)
      }
    case .float32:
      rgbDataAsFloats = [Float32](unsafeData: outputTensor.data) ?? []
    default:
      print("Output tensor data type \(outputTensor.dataType) is unsupported for this example app.")
      return nil
    }
    
    let rgbData: [UInt8]
    if #available(iOS 13.0, *) {
      // Use faster, vectorized operations if available.
      rgbData = vDSP.floatingPointToInteger(vDSP.multiply(255, rgbDataAsFloats),
                                                integerType: UInt8.self,
                                                rounding: .towardNearestInteger)
    } else {
      // Fallback on earlier versions.
      rgbData = rgbDataAsFloats.map { UInt8($0 * 255) }
    }
    
    // Convert float array to StyleTransferOutput
    let image = convertRGBToImage(rgbData: rgbData, width: contentImageSize, height: contentImageSize) ?? UIImage()
    
    // Return the inference time and inference results.
    return Result<StyleTransferOutput>(elapsedTimeInMs: interval, inference: image)
  }

  private func convertRGBToImage(rgbData: [UInt8], width: Int, height: Int) -> UIImage? {
    // Set up local constants to define the raw image format.
    let bitsPerComponent = 8
    let componentsPerPixel = 3
    let bitsPerPixel = bitsPerComponent * componentsPerPixel
    
    // Check that the parameters and rgb data are valid.
    guard width > 0 && height > 0 else { return nil }
    guard rgbData.count == width * height * componentsPerPixel else { return nil }
    
    let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
    let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue)
    
    var rgbDataMutable = rgbData
    guard let providerRef = CGDataProvider(data: NSData(bytes: &rgbDataMutable,
                                                        length: rgbDataMutable.count * MemoryLayout<UInt8>.size))
      else { return nil }
    
    // Create the image with CoreGraphics.
    guard let cgImage = CGImage(
      width: width,
      height: height,
      bitsPerComponent: bitsPerComponent,
      bitsPerPixel: bitsPerPixel,
      bytesPerRow: width * MemoryLayout<UInt8>.size * componentsPerPixel,
      space: rgbColorSpace,
      bitmapInfo: bitmapInfo,
      provider: providerRef,
      decode: nil,
      shouldInterpolate: true,
      intent: .defaultIntent
      )
      else { return nil }
    
    // Convert to a UIImage.
    return UIImage(cgImage: cgImage)
  }
  
}
