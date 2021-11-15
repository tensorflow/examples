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

/// The metadata to bake into a TFLite model file.
struct ModelMetadata {
  /// The name of the model.
  var name = ""
  /// The description of the model.
  var description = ""
  /// The author of the model.
  var author = ""
  /// The version of the model.
  var version = "1.0"
  /// The license under which the model is released.
  var license = "Apache"

  /// Whether the metadata are complete enough to be baked into a TFLite model file.
  var isValid: Bool {
    !name.isEmpty && !description.isEmpty && !author.isEmpty && !version.isEmpty && !license.isEmpty
  }
}
