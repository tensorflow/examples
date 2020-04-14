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

class WordpieceTokenizerTest: XCTestCase {
  func testTokenize() {
    let vocabularyIDs = FileLoader.loadVocabularies(from: MobileBERT.vocabulary)
    let tokenizer = WordpieceTokenizer(with: vocabularyIDs)

    XCTAssertEqual(tokenizer.tokenize("meaningfully"), ["meaningful", "##ly"])
    XCTAssertEqual(tokenizer.tokenize("teacher"), ["teacher"])
  }

  func testTokenizerWithCustomvocabularyIDs() {
    let vocab = ["[UNK]", "[CLS]", "[SEP]", "want", "##want", "##ed", "wa", "un", "runn", "##ing"]
    var vocabularyIDs = [String: Int32]()
    for (index, string) in vocab.enumerated() {
      vocabularyIDs[string] = Int32(index)
    }
    let tokenizer = WordpieceTokenizer(with: vocabularyIDs)

    XCTAssertEqual(tokenizer.tokenize(""), [])
    XCTAssertEqual(
      tokenizer.tokenize("unwanted running"), ["un", "##want", "##ed", "runn", "##ing"])
    XCTAssertEqual(tokenizer.tokenize("unwantedX running"), ["[UNK]", "runn", "##ing"])
  }
}
