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

class StringExtensionTest: XCTestCase {
  func testCleaned() {
    let testExample1 = "This is an\rexample.\n"
    let testExample2 = testExample1 + "\u{0}"
    let testExample3 = testExample1 + "\u{fffd}"

    let expectedResult = "This is an example. "

    XCTAssertEqual(testExample1.cleaned(), expectedResult)
    XCTAssertEqual(testExample2.cleaned(), expectedResult)
    XCTAssertEqual(testExample3.cleaned(), expectedResult)
  }

  func testSplitByWhitespace() {
    let testExample = "Hi ,\n This is an example. "
    let expectedResult = ["Hi", ",", "This", "is", "an", "example."]
    let reversedResult = [String](expectedResult.reversed())

    XCTAssertNotEqual(testExample.splitByWhitespace(), reversedResult)
    XCTAssertEqual(testExample.splitByWhitespace(), expectedResult)
    XCTAssertEqual("       ".splitByWhitespace(), [])
    XCTAssertEqual("".splitByWhitespace(), [])
  }

  func testTokenizedWithPunctuation() {
    let testExample1 = "Hi,there."
    let expectedResult1 = ["Hi", ",", "there", "."]
    let reversedResult1 = [String](expectedResult1.reversed())

    XCTAssertEqual(testExample1.tokenizedWithPunctuation(), expectedResult1)
    XCTAssertNotEqual(testExample1.tokenizedWithPunctuation(), reversedResult1)

    let testExample2 = "I\'m \"Spider-Man\""  // Input: I'm "Spider-Man"
    let expectedResult2 = ["I", "\'", "m ", "\"", "Spider", "-", "Man", "\""]
    let reversedResult2 = [String](expectedResult2.reversed())

    XCTAssertNotEqual(testExample2.tokenizedWithPunctuation(), reversedResult2)
    XCTAssertEqual(testExample2.tokenizedWithPunctuation(), expectedResult2)
    XCTAssertEqual("".tokenizedWithPunctuation(), [])

  }
}
