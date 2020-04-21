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

/// Provides some functions that make it easy to classify the character.
extension UnicodeScalar {
  /// Whether `self` is a whitespace character.
  ///
  /// \t, \n, and \r are technically control characters but we treat them as whitespace since they
  /// are generally considered as such.
  var isWhitespaceForBert: Bool {
    switch self {
    case " ", "\t", "\n", "\r":
      return true
    default:
      return properties.generalCategory == .spaceSeparator
    }
  }

  /// Whether `self` is a control character.
  var isControlForBert: Bool {
    // These are technically control characters but we count them as whitespace characters.
    if isWhitespaceForBert {
      return false
    }

    switch properties.generalCategory {
    case .control, .format: return true
    default: return false
    }
  }

  /// Whether `self` should be removed for Bert tokenization.
  var shouldBeRemovedForBert: Bool {
    return self == UnicodeScalar(0) || self == UnicodeScalar(0xfffd)
  }

  /// Whether `self` is a punctuation character.
  ///
  /// We treat all non-letter/number ASCII as punctuation, except ASCII character 0 to 32.
  /// Characters such as "^", "$", and "`" are not in the Unicode Punctuation class but we treat
  /// them as punctuation anyways, for consistency.
  var isPunctuationForBert: Bool {
    if isASCII && value > 32 && !properties.isAlphabetic && properties.numericType == nil {
      return true
    }
    switch properties.generalCategory {
    case .closePunctuation,
      .connectorPunctuation,
      .dashPunctuation,
      .finalPunctuation,
      .initialPunctuation,
      .openPunctuation,
      .otherPunctuation:
      return true
    default:
      return false
    }
  }
}
