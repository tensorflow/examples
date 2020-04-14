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

import UIKit

/// Data set to run the TensorFlow Lite model.
struct Dataset: Decodable {
  let title: String
  let content: String
  let questions: [String]

  /// Wrapper to decode json file into `Decodable` struct.
  static func load<T: Decodable>(_ file: File = MobileBERT.dataset) -> T {
    let data: Data

    guard let fileUrl = Bundle.main.url(forResource: file.name, withExtension: file.ext)
    else {
      fatalError("Couldn't find \(file.description) in main bundle.")
    }

    do {
      data = try Data(contentsOf: fileUrl)
    } catch {
      fatalError("Couldn't load \(file.description) from main bundle:\n\(error)")
    }

    do {
      let decoder = JSONDecoder()
      return try decoder.decode(T.self, from: data)
    } catch {
      fatalError("Couldn't parse \(file.description) as \(T.self):\n\(error)")
    }
  }
}
