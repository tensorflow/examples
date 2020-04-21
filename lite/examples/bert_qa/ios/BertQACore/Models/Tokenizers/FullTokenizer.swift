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

/// Runs end-to-end tokenization.
///
/// Name of functions and variables are from
/// [google-research/bert](https://github.com/google-research/bert/blob/d66a146741588fb208450bde15aa7db143baaa69/tokenization.py#L161).
class FullTokenizer {
  let basicTokenizer: BasicTokenizer
  let wordpieceTokenizer: WordpieceTokenizer
  /// A mapping of strings to IDs, where the IDs come from
  let vocabularyIDs: [String: Int32]

  /// Initialize `Fulltokenizer`.
  /// - Parameters
  ///   - vocabularyIDs: A mapping of strings to IDs, where the IDs come from the number order of
  ///   vocabulary from the vocabulary file.
  ///   - isCaseInsensitive: `true` if the tokenizer ignores the case.
  init(with vocabularyIDs: [String: Int32], isCaseInsensitive: Bool) {
    self.vocabularyIDs = vocabularyIDs
    basicTokenizer = BasicTokenizer(isCaseInsensitive: isCaseInsensitive)
    wordpieceTokenizer = WordpieceTokenizer(with: vocabularyIDs)
  }

  func tokenize(_ text: String) -> [String] {
    return basicTokenizer.tokenize(text).flatMap { wordpieceTokenizer.tokenize($0) }
  }

  func convertToIDs(tokens: [String]) -> [Int32] {
    return tokens.compactMap { vocabularyIDs[$0] }
  }
}
