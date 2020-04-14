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

/// Runs basic tokenization such as punctuation spliting, lower casing.
///
/// Name of functions and variables are from
/// [google-research/bert]( https://github.com/google-research/bert/blob/d66a146741588fb208450bde15aa7db143baaa69/tokenization.py#L185).
struct BasicTokenizer {
  let isCaseInsensitive: Bool

  /// Constructs a BasicTokenizer.
  ///
  /// - Parameter isCaseInsensitive: Whether to lower case the input.
  init(isCaseInsensitive: Bool) {
    self.isCaseInsensitive = isCaseInsensitive
  }

  /// Tokenizes a piece of text.
  ///
  /// - Parameter text: Text to be tokenized.
  func tokenize(_ text: String) -> [String] {
    var cleanedText = text.cleaned()
    if isCaseInsensitive {
      cleanedText = cleanedText.lowercased()
    }

    return cleanedText.splitByWhitespace().flatMap { $0.tokenizedWithPunctuation() }
  }
}
