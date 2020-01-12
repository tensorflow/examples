//
//  TextClassificationTests.swift
//  TextClassificationTests
//
//  Created by Khurram Shehzad on 06/01/2020.
//  Copyright © 2020 Khurram Shehzad. All rights reserved.
//

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
