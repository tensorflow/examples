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

import SwiftUI

/// Displays a form to fill information about the model to create.
struct ModelCreationView: View {
  /// The closure to call once the Model object will be created.
  let completion: (Model?) -> Void
  /// The metadata of the model to create.
  @State private var modelMetadata = ModelMetadata()

  var body: some View {
    Form {
      Field(key: "Model Name", value: $modelMetadata.name)
      Field(key: "Description", value: $modelMetadata.description)
      Field(key: "Author", value: $modelMetadata.author)
      Field(key: "Version", value: $modelMetadata.version)
      Field(key: "License", value: $modelMetadata.license)
    }
    .navigationBarTitle(Text("New Model"), displayMode: .inline)
    .toolbar {
      ToolbarItem(placement: .primaryAction) {
        NavigationLink(
          "Next", destination: AlbumSelector(modelMetadata: modelMetadata, completion: completion)
        )
        .accessibilityHint(
          "Proceeds to the next model creation step. Dimmed until all info is filled."
        )
        .disabled(!modelMetadata.isValid)
      }
      ToolbarItem(placement: .cancellationAction) {
        Button("Cancel") {
          completion(nil)
        }
      }
    }
  }
}
