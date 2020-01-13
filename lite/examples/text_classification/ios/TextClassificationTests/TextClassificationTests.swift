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
@testable import TextClassification

class TextClassificationTests: XCTestCase {

private var client: TextClassificationnClient?

override func setUp() {
  client = TextClassificationnClient(modelFileInfo: modelFileInfo, labelsFileInfo: labelsFileInfo, vocabFileInfo: vocabFileInfo)
}

func testLoadModel() {
  XCTAssertNotNil(client)
}

func testLoadDictionaryTest() {
  XCTAssertEqual(client?.dictionary["<PAD>"], 0)
  XCTAssertEqual(client?.dictionary["<START>"], 1)
  XCTAssertEqual(client?.dictionary["<UNKNOWN>"], 2)
  XCTAssertEqual(client?.dictionary["the"], 4)
}

func testLoadLabels() {
  XCTAssertEqual(client?.labels[0], "Negative")
  XCTAssertEqual(client?.labels[1], "Positive")
}
  
func testInputPreprocessing() {
  let clientOutput = client?.tokenizeInputText(text: "hello,world!")
  var expectOutput = [[Float](repeating: 0, count: 256)]
  expectOutput[0][0] = 1; // Index for <START>
  expectOutput[0][1] = 4825; // Index for "hello".
  expectOutput[0][2] = 182; // Index for "world".
  XCTAssertTrue(clientOutput![0].elementsEqual(expectOutput[0]))
}

func testPredictPositive() {
  let positiveText = client?.classify(text: "This is an interesting film. My family and I all liked it very much.")[0]
  XCTAssertEqual(positiveText?.title, "Positive")
  XCTAssertTrue(positiveText!.confidence > Float(0.55))
}
  
func testPredictNegative() {
  let negativeText = client?.classify(text: "This film cannot be worse. It is way too boring.")[0]
  XCTAssertEqual(negativeText?.title, "Negative")
  XCTAssertTrue(negativeText!.confidence > Float(0.6))
}
} // class TextClassificationTests
