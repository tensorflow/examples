// Copyright 2021 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import Foundation

/// Manages the persistence of the list of models to a file on disk.
class ModelData: ObservableObject {
  /// The list of loaded models.
  @Published var models: [Model] = []
  /// The URL on disk where the models are persisted.
  static var fileURL: URL {
    FileManager.documentDirectory.appendingPathComponent("Models.data")
  }

  /// Loads synchronously the list of models in memory.
  func load() throws {
    guard FileManager.default.fileExists(atPath: Self.fileURL.path) else { return }
    let data = try Data(contentsOf: Self.fileURL)
    models = try JSONDecoder().decode([Model].self, from: data)
  }

  /// Saves synchronously the list of models to disk.
  func save() throws {
    try JSONEncoder().encode(models).write(to: Self.fileURL)
  }
}
