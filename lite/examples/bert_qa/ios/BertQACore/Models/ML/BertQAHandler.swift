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

import Foundation
import TensorFlowLite
import os

/// Handles Bert model with TensorFlow Lite.
final class BertQAHandler {
  private let interpreter: Interpreter

  private let dic: [String: Int32]
  private let tokenizer: FullTokenizer

  init(
    with modelFile: File = MobileBERT.model,
    threadCount: Int = InterpreterOptions.threadCount.defaultValue
  ) throws {
    os_log(
      "[BertQAHandler] Initialize interpreter with %d thread(s).", threadCount)
    // Load dictionary from vocabulary file.
    dic = FileLoader.loadVocabularies(from: MobileBERT.vocabulary)

    // Initialize `FullTokenizer` with given `dic`.
    tokenizer = FullTokenizer(with: dic, isCaseInsensitive: MobileBERT.doLowerCase)

    // Construct the path to the model file.
    guard
      let modelPath = Bundle(for: type(of: self)).path(
        forResource: modelFile.name,
        ofType: modelFile.ext)
    else {
      fatalError("Failed to load the model file: \(modelFile.description)")
    }

    // Specify the options for the `Interpreter`.
    var options = Interpreter.Options()
    options.threadCount = threadCount

    // Create the `Interpreter`.
    interpreter = try Interpreter(modelPath: modelPath, options: options)

    // Initialize input and output `Tensor`s.
    try interpreter.allocateTensors()

    // Get allocated input `Tensor`s.
    let inputIdsTensor: Tensor
    let inputMaskTensor: Tensor
    let segmentIdsTensor: Tensor

    inputIdsTensor = try interpreter.input(at: 0)
    inputMaskTensor = try interpreter.input(at: 1)
    segmentIdsTensor = try interpreter.input(at: 2)

    // Get allocated output `Tensor`s.
    let endLogitsTensor: Tensor
    let startLogitsTensor: Tensor

    endLogitsTensor = try interpreter.output(at: 0)
    startLogitsTensor = try interpreter.output(at: 1)

    // Check if input and output `Tensor`s are in the expected formats.
    guard
      inputIdsTensor.shape.dimensions == MobileBERT.inputDimension
        && inputMaskTensor.shape.dimensions == MobileBERT.inputDimension
        && segmentIdsTensor.shape.dimensions == MobileBERT.inputDimension
    else {
      fatalError("Unexpected model: Input Tensor shape")
    }

    guard
      endLogitsTensor.shape.dimensions == MobileBERT.outputDimension
        && startLogitsTensor.shape.dimensions == MobileBERT.outputDimension
    else {
      fatalError("Unexpected model: Output Tensor shape")
    }
  }

  func run(query: String, content: String) -> Result? {
    // MARK: - Preprocessing
    let (features, contentData) = preprocessing(query: query, content: content)

    // Convert input arrays to `Data` type.
    os_log("[BertQAHandler] Setting inputs..")
    let inputIdsData = Data(copyingBufferOf: features.inputIds)
    let inputMaskData = Data(copyingBufferOf: features.inputMask)
    let segmentIdsData = Data(copyingBufferOf: features.segmentIds)

    // MARK: - Inferencing
    let inferenceStartTime = Date()

    let endLogitsTensor: Tensor
    let startLogitsTensor: Tensor

    do {
      // Assign input `Data` to the `interpreter`.
      try interpreter.copy(inputIdsData, toInputAt: 0)
      try interpreter.copy(inputMaskData, toInputAt: 1)
      try interpreter.copy(segmentIdsData, toInputAt: 2)

      // Run inference by invoking the `Interpreter`.
      os_log("[BertQAHandler] Runing inference..")
      try interpreter.invoke()

      // Get the output `Tensor` to process the inference results
      endLogitsTensor = try interpreter.output(at: 0)
      startLogitsTensor = try interpreter.output(at: 1)
    } catch let error {
      os_log(
        "[BertQAHandler] Failed to invoke the interpreter with error: %s",
        type: .error,
        error.localizedDescription)
      return nil
    }

    let inferenceTime = Date().timeIntervalSince(inferenceStartTime) * 1000

    // MARK: - Postprocessing
    os_log("[BertQAHandler] Getting answer..")
    let answers = postprocessing(
      startLogits: startLogitsTensor.data.toArray(type: Float32.self),
      endLogits: endLogitsTensor.data.toArray(type: Float32.self),
      contentData: contentData)
    os_log("[BertQAHandler] Finished.")

    guard let answer = answers.first else {
      return nil
    }
    return Result(answer: answer, inferenceTime: inferenceTime)
  }

  // MARK: - Private functions

  /// Tokenizes input query and content to `InputFeatures` and make `ContentData` to find the
  /// original string in the content.
  ///
  /// - Parameters:
  ///   - query: Input query to run the model.
  ///   - content: Input content to run the model.
  /// - Returns: A tuple of `InputFeatures` and `ContentData`.
  private func preprocessing(query: String, content: String) -> (InputFeatures, ContentData) {
    var queryTokens = tokenizer.tokenize(query)
    queryTokens = Array(queryTokens.prefix(MobileBERT.maxQueryLen))

    let contentWords = content.splitByWhitespace()
    var contentTokenIdxToWordIdxMapping = [Int]()
    var contentTokens = [String]()
    for (i, token) in contentWords.enumerated() {
      tokenizer.tokenize(token).forEach { subToken in
        contentTokenIdxToWordIdxMapping.append(i)
        contentTokens.append(subToken)
      }
    }

    // -3 accounts for [CLS], [SEP] and [SEP].
    let maxContentLen = MobileBERT.maxSeqLen - queryTokens.count - 3
    contentTokens = Array(contentTokens.prefix(maxContentLen))

    var tokens = [String]()
    var segmentIds = [Int32]()

    // Map token index to original index (in feature.origTokens).
    var tokenIdxToWordIdxMapping = [Int: Int]()

    // Start of generating the `InputFeatures`.
    tokens.append("[CLS]")
    segmentIds.append(0)

    // For query input.
    queryTokens.forEach {
      tokens.append($0)
      segmentIds.append(0)
    }

    // For separation.
    tokens.append("[SEP]")
    segmentIds.append(0)

    // For text input.
    for (i, docToken) in contentTokens.enumerated() {
      tokens.append(docToken)
      segmentIds.append(1)
      tokenIdxToWordIdxMapping[tokens.count] = contentTokenIdxToWordIdxMapping[i]
    }

    // For ending mark.
    tokens.append("[SEP]")
    segmentIds.append(1)

    var inputIds = tokenizer.convertToIDs(tokens: tokens)
    var inputMask = [Int32](repeating: 1, count: inputIds.count)

    while inputIds.count < MobileBERT.maxSeqLen {
      inputIds.append(0)
      inputMask.append(0)
      segmentIds.append(0)
    }

    let inputFeatures = InputFeatures(
      inputIds: inputIds,
      inputMask: inputMask,
      segmentIds: segmentIds)
    let contentData = ContentData(
      contentWords: contentWords,
      tokenIdxToWordIdxMapping: tokenIdxToWordIdxMapping,
      originalContent: content)
    return (inputFeatures, contentData)
  }

  /// Get a list of most possible `Answer`s up to `Model.predictAnsNum`.
  ///
  /// - Parameters:
  ///   - startLogits: List of `Logit` if the index can be a start token index of an answer.
  ///   - endLogits: List of `Logit` if the index can be a end token index of an answer.
  ///   - features: `InputFeatures` used to run the model.
  /// - Returns: List of `Answer`s.
  private func postprocessing(
    startLogits: [Float],
    endLogits: [Float],
    contentData: ContentData
  ) -> [Answer] {
    // Get the candidate start/end indexes of answer from `startLogits` and `endLogits`.
    let startIndexes = candidateAnswerIndexes(from: startLogits)
    let endIndexes = candidateAnswerIndexes(from: endLogits)

    // Make list which stores prediction and its range to find original results and filter invalid
    // pairs.
    let candidates: [Prediction] = startIndexes.flatMap { start in
      endIndexes.compactMap { end -> Prediction? in
        // Initialize logit struct with given indexes.
        guard
          let prediction = Prediction(
            logit: startLogits[start] + endLogits[end],
            start: start, end: end,
            tokenIdxToWordIdxMapping: contentData.tokenIdxToWordIdxMapping)
        else {
          return nil
        }
        return prediction
      }
    }.sorted { $0.logit > $1.logit }

    // Extract firstmost `Model.predictAnsNum` of predictions and calculate score from logits array
    // with softmax.
    let scores = softmaxed(Array(candidates.prefix(MobileBERT.predictAnsNum)))

    // Return answer list.
    return scores.compactMap { score in
      guard
        let excerpted = contentData.excerptWords(from: score.range)
      else {
        return nil
      }
      return Answer(text: excerpted, score: score)
    }
  }

  /// Get the `Model.prediectAnsNum` number of indexes of the candidate answers from given logit
  /// list.
  ///
  /// - Parameter from: The array of logits.
  /// - Returns: `Model.predictAnsNum` number of indexes.
  private func candidateAnswerIndexes(from logits: [Float]) -> [Int] {
    return logits.prefix(MobileBERT.maxSeqLen)
      .enumerated()
      .sorted { $0.element > $1.element }
      .prefix(MobileBERT.predictAnsNum)
      .map { $0.offset }
  }

  /// Compute softmax probability score over raw logits.
  ///
  /// - Parameter predictions: Array of logit and it range sorted by the logit value in decreasing
  ///   order.
  private func softmaxed(_ predictions: [Prediction]) -> [Score] {
    // Find maximum logit value.
    guard let maximumLogit = predictions.first?.logit else {
      return []
    }

    // Calculate numerator array of the softmaxed values and its sum.
    let numerators: [(Float, Prediction)] = predictions.map { prediction in
      let numerator = exp(prediction.logit - maximumLogit)
      return (numerator, prediction)
    }

    let sum: Float = numerators.reduce(0) { $0 + $1.0 }

    return numerators.compactMap { (numerator, prediction) in
      Score(value: numerator / sum, range: prediction.wordRange, logit: prediction.logit)
    }
  }
}
