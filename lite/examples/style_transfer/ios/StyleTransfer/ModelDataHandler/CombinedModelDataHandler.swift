//
//  CombinedModelDataHandler.swift
//  StyleTransfer
//
//  Created by Ivan Cheung on 3/28/20.
//  Copyright © 2020 Ivan Cheung. All rights reserved.
//

import TensorFlowLite

/// This class handles all data preprocessing and makes calls to perform style transfer on a given frame
/// by invoking the `Interpreter`. It uses two internal model handlers to do this.
/// First, it invokes the style predictor model to generate style bottleneck.
/// Next, it applies the model bottleneck to the given frame using the style transfer model.
class CombinedModelDataHandler: ModelDataHandling {
  // MARK: - Internal Properties
  var style: Style = .style0
  
  let threadCount: Int
  let threadCountLimit = 10
  
  // MARK: - Private Properties
  private let stylePredictorModelDataHandler: StylePredictorModelDataHandler
  private let styleTransferModelDataHandler: StyleTransferModelDataHandler
  
  // Cache predicted style bottlenecks
  private var styleBottleneckCache = [Style: StyleBottleneck]()
    
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
    let styleBottleneckResult: Result<StyleBottleneck>?
    if let cachedStyleBottleneck = retrieveStyleBottlenextFromCache(style: style) {
      styleBottleneckResult = cachedStyleBottleneck
    } else {
      styleBottleneckResult = stylePredictorModelDataHandler.runModel(input: style)
      styleBottleneckCache[style] = styleBottleneckResult?.inference
    }
    
    guard let bottleneckResult = styleBottleneckResult else {
      return nil
    }
    
    guard let imageResult = styleTransferModelDataHandler.runModel(input: StyleTransferInput(
      styleBottleneck: bottleneckResult.inference,
      pixelBuffer: pixelBuffer)) else { return nil }
    
    let elapsedTimeInMs = bottleneckResult.elapsedTimeInMs + imageResult.elapsedTimeInMs

    return Result<UIImage>(elapsedTimeInMs: elapsedTimeInMs, inference: imageResult.inference)
  }
  
  // MARK: - Private Methods
  private func retrieveStyleBottlenextFromCache(style: Style) -> Result<StyleBottleneck>? {
    return styleBottleneckCache[style].map { Result(elapsedTimeInMs: 0, inference: $0) }
  }
}
