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

@testable import BertQA_UIKit

class BasicTokenizerTest: XCTestCase {
  func testTokenize() {
    let tokenizer = BasicTokenizer(isCaseInsensitive: false)
    let testInput1 = "  Hi, This\tis an example.\n"
    let expectedResult1 = ["Hi", ",", "This", "is", "an", "example", "."]

    XCTAssertEqual(tokenizer.tokenize(testInput1), expectedResult1)

    let testInput2 = "Hello,How are you?"
    let expectedResult2 = ["Hello", ",", "How", "are", "you", "?"]

    XCTAssertEqual(tokenizer.tokenize(testInput2), expectedResult2)
  }
}
