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

/// Manages content data to excerpt answer string with a token index.
struct ContentData {
  let contentWords: [String]
  let tokenIdxToWordIdxMapping: [Int: Int]
  let originalContent: String

  /// Find and excerpt the original string with given index.
  ///
  /// - Parameters:
  ///   - feature: `InputFeature` of the query.
  ///   - range: Range of token ID of the string to find.
  /// - Returns: Returns the original string with given range. `nil` if  the index is not correct.
  func excerptWords(from range: ClosedRange<Int>) -> Excerpt? {
    let pattern: String = contentWords[range].map {
      NSRegularExpression.escapedPattern(for: $0)
    }.joined(separator: "\\s+")

    let compareOptions: NSString.CompareOptions
    if range.count == 1 {
      compareOptions = []
    } else {
      compareOptions = .regularExpression
    }

    guard
      let range = originalContent.range(of: pattern, options: compareOptions)
    else {
      return nil
    }
    return Excerpt(value: String(originalContent[range]), range: range)
  }
}
