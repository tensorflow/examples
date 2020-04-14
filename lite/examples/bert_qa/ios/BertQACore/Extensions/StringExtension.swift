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
import os

/// Helper functions used for tokenizing.
extension String {
  /// Performs invalid character removal and whitespace cleanup on text.
  ///
  /// Replaces all whitespace code points with spaces and control characters including \t, \n, \r.
  ///
  /// - Returns: Cleaned text.
  func cleaned() -> String {
    return String(
      // Normalize string to NFC(Normalization Form Canonical Composition).
      self.precomposedStringWithCanonicalMapping
        .unicodeScalars.compactMap { unicodeScalar in
          if unicodeScalar.isWhitespaceForBert {
            return " "
          } else if !unicodeScalar.isControlForBert && !unicodeScalar.shouldBeRemovedForBert {
            return Character(unicodeScalar)
          }
          return nil
        })

  }

  /// Splits this string on whitespace.
  func splitByWhitespace() -> [String] {
    // Normalize string to NFC(Normalization Form Canonical Composition).
    return self.precomposedStringWithCanonicalMapping
      .unicodeScalars.split { $0.isWhitespaceForBert }.map { String($0) }
  }

  /// Tokenizes this string into word and punctuation tokens.
  ///
  /// For example:
  /// ```
  /// input: "Hi,there."
  /// output: ["Hi", ",", "there", "."]
  /// ```
  /// ```
  /// input: "Hi, there.\n"
  /// output: ["Hi", ",", " there", ".", "\n"]
  /// ```
  func tokenizedWithPunctuation() -> [String] {
    var tokens = [String]()
    var currentToken = ""
    // Normalize string to NFC(Normalization Form Canonical Composition).
    self.precomposedStringWithCanonicalMapping
      .unicodeScalars.forEach { unicode in
        if unicode.isPunctuationForBert {
          if !currentToken.isEmpty {
            // Add current token before the punctuation mark to the list of tokens.
            tokens.append(currentToken)
          }
          tokens.append(String(unicode))
          currentToken = ""
        } else {
          // As it is not a punctuation mark, keep building current token.
          currentToken += String(unicode)
        }
      }

    if !currentToken.isEmpty {
      tokens.append(currentToken)
    }
    return tokens
  }
}
