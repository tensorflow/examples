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

/// Describes a TFLite model stored locally.
struct Model: Codable, Equatable {
  /// The name of the model.
  let name: String
  /// The number of labels in its labelmap.
  let labelsCount: Int
  /// The estimated size of the model file on disk.
  let size: Int64?
}

extension Model {
  /// Returns the URL where to store a new Tflite model file.
  ///
  /// - Parameter name: The name of the future model.
  static func makeURLForNewModel(named name: String) -> URL {
    url(modelNamed: name)
  }

  /// The `URL` where the associated TFLite model file is stored.
  var url: URL {
    Self.url(modelNamed: name)
  }

  /// Returns the URL of the TFLite model file associated with a model named `name`.
  ///
  /// - Parameter name: The name of the model.
  private static func url(modelNamed name: String) -> URL {
    FileManager.documentDirectory.appendingPathComponent(name).appendingPathExtension("tflite")
  }
}
