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

class BertQAHandlerTest: XCTestCase {
  /// Test BertQA handler with ASCII code only content & question.
  func testBertQAHandler() {
    let bertQA: BertQAHandler
    do {
      bertQA = try BertQAHandler()

      let content =
        "TensorFlow is a free and open-source software library for dataflow and "
        + "differentiable programming across a range of tasks. It is a symbolic math library, and "
        + "is also used for machine learning applications such as neural networks. It is used for "
        + "both research and production at Google. TensorFlow was developed by the Google Brain "
        + "team for internal Google use. It was released under the Apache License 2.0 on November "
        + "9, 2015."

      let question1 = "What is TensorFlow"
      let answer1 =
        "a free and open-source software library for dataflow and differentiable "
        + "programming across a range of tasks"
      if let result1 = bertQA.run(query: question1, content: content) {
        XCTAssert(result1.answer.text.value.contains(answer1))
      } else {
        XCTFail("Failed to run BertQA with \(question1).")
      }

      let question2 = "Who developed TensorFlow?"
      let answer2 = "Google Brain team"
      if let result2 = bertQA.run(query: question2, content: content) {
        XCTAssert(result2.answer.text.value.contains(answer2))
      } else {
        XCTFail("Failed to run BertQA with \(question2).")
      }

      let question3 = "When was TensorFlow released?"
      let answer3 = "November 9, 2015"
      if let result3 = bertQA.run(query: question3, content: content) {
        XCTAssert(result3.answer.text.value.contains(answer3))
      } else {
        XCTFail("Failed to run BertQA with \(question3).")
      }

      let question4 = "What is TensorFlow used for?"
      let answer4 =
        "symbolic math library, and is also used for machine learning applications such as neural "
        + "networks"
      if let result4 = bertQA.run(query: question4, content: content) {
        XCTAssert(result4.answer.text.value.contains(answer4))
      } else {
        XCTFail("Failed to run BertQA with \(question4).")
      }

      let question5 = "How is TensorFlow used in Google?"
      let answer5 = "both research and production"
      if let result5 = bertQA.run(query: question5, content: content) {
        XCTAssert(result5.answer.text.value.contains(answer5))
      } else {
        XCTFail("Failed to run BertQA with \(question5).")
      }

      let question6 = "Which license does TensorFlow use?"
      let answer6 = "Apache License 2.0"
      if let result6 = bertQA.run(query: question6, content: content) {
        XCTAssert(result6.answer.text.value.contains(answer6))
      } else {
        XCTFail("Failed to run BertQA with \(question6).")
      }

    } catch let error {
      XCTFail(error.localizedDescription)
    }
  }

  /// Test BertQA handler with a content & question including unicode.
  func testBertQAHandlerWithUnicode() {
    let bertQA: BertQAHandler
    do {
      bertQA = try BertQAHandler()

      let content =
        "Nikola Tesla (Serbian Cyrillic: \u{041d}\u{0438}\u{043a}\u{043e}\u{043b}"
        + "\u{0430} \u{0422}\u{0435}\u{0441}\u{043b}\u{0430}; 10 July 1856 \u{2013} 7 January 1943) "
        + "was a Serbian American inventor, electrical engineer, mechanical engineer, physicist, and "
        + "futurist best known for his contributions to the design of the modern alternating current "
        + "(AC) electricity supply system."

      let question1 = "What is Tesla's home country?"
      let answer1 = "Serbian"
      if let result1 = bertQA.run(query: question1, content: content) {
        XCTAssert(result1.answer.text.value.contains(answer1))
      } else {
        XCTFail()
        XCTFail("Failed to run BertQA with \(question1).")
      }

      let question2 = "What was Nikola Tesla's ethnicity?"
      let answer2 = "Serbian"
      if let result2 = bertQA.run(query: question2, content: content) {
        XCTAssert(result2.answer.text.value.contains(answer2))
      } else {
        XCTFail("Failed to run BertQA with \(question2).")
      }

      let question3 = "What does AC stand for?"
      let answer3 = "alternating current"
      if let result3 = bertQA.run(query: question3, content: content) {
        XCTAssert(result3.answer.text.value.contains(answer3))
      } else {
        XCTFail("Failed to run BertQA with \(question3).")
      }

      let question4 = "When was Tesla born?"
      let answer4 = "10 July 1856"
      if let result4 = bertQA.run(query: question4, content: content) {
        XCTAssert(result4.answer.text.value.contains(answer4))
      } else {
        XCTFail("Failed to run BertQA with \(question4).")
      }

      let question5 = "In what year did Tesla die?"
      let answer5 = "1943"
      if let result5 = bertQA.run(query: question5, content: content) {
        XCTAssert(result5.answer.text.value.contains(answer5))
      } else {
        XCTFail("Failed to run BertQA with \(question5).")
      }

    } catch let error {
      XCTFail(error.localizedDescription)
    }
  }
}
