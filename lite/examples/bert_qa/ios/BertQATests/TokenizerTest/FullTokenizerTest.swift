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

import XCTest
import os

@testable import BertQA_UIKit

class FullTokenizerTest: XCTestCase {
  func testTokenize() {
    let vocabularyIDs = FileLoader.loadVocabularies(from: MobileBERT.vocabulary)
    let tokenizer = FullTokenizer(with: vocabularyIDs, isCaseInsensitive: true)
    XCTAssertEqual(tokenizer.tokenize(""), [])

    let testInput1 = "Good morning, I'm your teacher.\n"
    let expectedResult1 = ["good", "morning", ",", "i", "'", "m", "your", "teacher", "."]
    XCTAssertEqual(tokenizer.tokenize(testInput1), expectedResult1)

    let testInput2 = "Nikola Tesla\t(Serbian Cyrillic: 10 July 1856 ~ 7 January 1943)"
    let expectedResult2 = [
      "nikola", "tesla", "(", "serbian", "cyrillic", ":", "10", "july", "1856", "~", "7", "january",
      "1943", ")",
    ]
    XCTAssertEqual(tokenizer.tokenize(testInput2), expectedResult2)
  }

  func testConvertTokensToIds() {
    let vocabularyIDs = FileLoader.loadVocabularies(from: MobileBERT.vocabulary)
    let tokenizer = FullTokenizer(with: vocabularyIDs, isCaseInsensitive: true)
    let testInput = ["good", "morning", ",", "i", "'", "m", "your", "teacher", "."]
    let expectedResult: [Int32] = [2204, 2851, 1010, 1045, 1005, 1049, 2115, 3836, 1012]

    XCTAssertEqual(tokenizer.convertToIDs(tokens: testInput), expectedResult)
  }
}
