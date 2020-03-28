//
//  CombinedModelDataHandler.swift
//  ImageClassification
//
//  Created by Ivan Cheung on 3/28/20.
//  Copyright © 2020 Y Media Labs. All rights reserved.
//

import TensorFlowLite

/// This class handles all data preprocessing and makes calls to run inference on a given frame
/// by invoking the `Interpreter`. It then formats the inferences obtained and returns the top N
/// results for a successful inference.
class CombinedModelDataHandler: ModelDataHandling {
  // MARK: - Internal Properties
  var inputWidth: Int { styleTransferModelDataHandler.contentImageSize }
  var inputHeight: Int { styleTransferModelDataHandler.contentImageSize }
  let threadCount: Int
  let threadCountLimit = 10
  
  // MARK: - Private Properties
  let stylePredictorModelDataHandler: StylePredictorModelDataHandler
  let styleTransferModelDataHandler: StyleTransferModelDataHandler
  
  // TODO: Cache last style predictor to use if not changed
    
  // MARK: - Initialization

    
  /// A failable initializer for `ModelDataHandler`. A new instance is created if the model and
  /// labels files are successfully loaded from the app's main bundle. Default `threadCount` is 1.
  init?(threadCount: Int = 1) {
    guard let stylePredictorModelDataHandler = StylePredictorModelDataHandler(threadCount: threadCount),
    let styleTransferModelDataHandler = StyleTransferModelDataHandler(threadCount: threadCount) else { return nil }
    
    self.stylePredictorModelDataHandler = stylePredictorModelDataHandler
    self.styleTransferModelDataHandler = styleTransferModelDataHandler
    
    self.threadCount = threadCount
  }

  // MARK: - Internal Methods

  /// Performs image preprocessing, invokes the `Interpreter`, and processes the inference results.
  func runModel(input pixelBuffer: CVPixelBuffer) -> Result<UIImage>? {
    guard let styleBottleneckResult = stylePredictorModelDataHandler.runModel(input: .style24) else {
      return nil
    }
    
    guard let imageResult = styleTransferModelDataHandler.runModel(input: StyleTransferInput(
      styleBottleneck: styleBottleneckResult.inference,
      pixelBuffer: pixelBuffer)) else { return nil }
    
    let elapsedTimeInMs = styleBottleneckResult.elapsedTimeInMs + imageResult.elapsedTimeInMs
    
    print("""
          Style prediction: \(styleBottleneckResult.elapsedTimeInMs)
          Style transfer: \(imageResult.elapsedTimeInMs)
          Total: \(elapsedTimeInMs)\n
          """)
    
    return Result<UIImage>(elapsedTimeInMs: elapsedTimeInMs, inference: imageResult.inference)
  }
}
