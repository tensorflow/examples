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

struct Result {
  let answer: Answer
  let inferenceTime: Double
  var description: String {
    """
    Inference time: \(String(format: "%.2lf ms", inferenceTime))
    Score: \(String(format: "%.2lf", answer.score.value))
    """
  }
}

struct Answer {
  let text: Excerpt
  let score: Score
}

/// Stores exceprted text and its range from the original text.
struct Excerpt {
  let value: String
  let range: Range<String.Index>
}

/// Stores probability score of given range of words in the original content.
struct Score {
  let value: Float
  /// Score's range of original word.
  let range: ClosedRange<Int>
  /// Logit value of this score.
  let logit: Float
}

/// Stores logit value and its range in token and word list.
struct Prediction {
  /// Logit value.
  let logit: Float
  /// Logit's range of result token.
  let tokenRange: ClosedRange<Int>
  /// Logit's range of original word.
  let wordRange: ClosedRange<Int>

  init?(logit: Float, start: Int, end: Int, tokenIdxToWordIdxMapping: [Int: Int]) {
    self.logit = logit
    guard start <= end else { return nil }
    self.tokenRange = start...end

    guard
      let wordRange = Prediction.convert(from: tokenRange, with: tokenIdxToWordIdxMapping)
    else {
      return nil
    }
    self.wordRange = wordRange
  }

  private static func convert(from tokenRange: ClosedRange<Int>, with map: [Int: Int])
    -> ClosedRange<Int>?
  {
    guard
      tokenRange.count <= MobileBERT.maxAnsLen,
      let start = tokenRange.first,
      let end = tokenRange.last,
      let startIndex = map[start + MobileBERT.outputOffset],
      let endIndex = map[end + MobileBERT.outputOffset]
    else {
      return nil
    }

    guard startIndex <= endIndex else { return nil }
    return startIndex...endIndex
  }
}

extension Prediction: Equatable {
  static func == (lhs: Prediction, rhs: Prediction) -> Bool {
    return lhs.logit == rhs.logit && lhs.tokenRange == rhs.tokenRange
  }
}
