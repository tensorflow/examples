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

class UnicodeScalarExtensionTest: XCTestCase {

  func testIsWhitespaceForBert() {
    XCTAssertTrue(UnicodeScalar(" ").isWhitespaceForBert)
    XCTAssertTrue(UnicodeScalar("\t").isWhitespaceForBert)
    XCTAssertTrue(UnicodeScalar("\r").isWhitespaceForBert)
    XCTAssertTrue(UnicodeScalar("\n").isWhitespaceForBert)
    XCTAssertTrue(UnicodeScalar(0x00A0).isWhitespaceForBert)

    XCTAssertFalse(UnicodeScalar("A").isWhitespaceForBert)
    XCTAssertFalse(UnicodeScalar("-").isWhitespaceForBert)
  }

  func testIsControlForBert() {
    XCTAssertTrue(UnicodeScalar(0x0005).isControlForBert)

    XCTAssertFalse(UnicodeScalar("A").isControlForBert)
    XCTAssertFalse(UnicodeScalar(" ").isControlForBert)
    XCTAssertFalse(UnicodeScalar("\t").isControlForBert)
    XCTAssertFalse(UnicodeScalar("\r").isControlForBert)
    XCTAssertFalse(UnicodeScalar("\u{1F4A9}").isControlForBert)
  }

  func testIsPunctuationForBert() {
    XCTAssertTrue(UnicodeScalar("-").isPunctuationForBert)
    XCTAssertTrue(UnicodeScalar("$").isPunctuationForBert)
    XCTAssertTrue(UnicodeScalar("`").isPunctuationForBert)
    XCTAssertTrue(UnicodeScalar(".").isPunctuationForBert)

    XCTAssertFalse(UnicodeScalar("A").isPunctuationForBert)
    XCTAssertFalse(UnicodeScalar(" ").isPunctuationForBert)
  }
}
