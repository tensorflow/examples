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
import os

class FileLoader {
  /// Loads a vocabulary file into a dictionary of vocabulary to its ID.
  ///
  /// - Parameter file: `File` of a vocabulary.
  /// - Returns: Vocabulary IDs from given `file` data.
  static func loadVocabularies(from file: File) -> [String: Int32] {
    guard
      let path = Bundle(for: FileLoader.self).path(forResource: file.name, ofType: file.ext)
    else {
      fatalError("Cannot read the file: \(file.description)")
    }

    var vocabularyIDs = [String: Int32]()
    do {
      let data = try String(contentsOfFile: path, encoding: .utf8)
      for (index, string) in data.components(separatedBy: .newlines).enumerated() {
        vocabularyIDs[string] = Int32(index)
      }
    } catch {
      os_log("%s", type: .error, error.localizedDescription)
    }
    return vocabularyIDs
  }
}
