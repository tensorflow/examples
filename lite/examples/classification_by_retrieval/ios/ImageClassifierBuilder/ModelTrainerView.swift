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

/// Displays a progress view while the model is being trained, and the result once trained.
///
/// This screen is part of the model creation UX.
struct ModelTrainerView: View {
  /// The metadata of the model to create.
  let modelMetadata: ModelMetadata
  /// The list of selected albums from the Photos Library.
  let albums: [Album]
  /// The closure to call once the Model object will be created.
  let completion: (Model?) -> Void
  /// The Model object associated to the model trained in this step.
  @State private var model: Model?

  var body: some View {
    VStack {
      if model == nil {
        Text("The model is being trained…")
          .padding()
        ProgressView()
      } else {
        Text("The model has been trained.")
          .padding()
        Image(systemName: "checkmark")
          .accessibility(hidden: true)
      }
    }
    .onAppear {
      // onAppear will strangely be called if the back button is tapped [1], on a new
      // ModelTrainerView whose albums are not set.
      // [1]: https://developer.apple.com/forums/thread/655338
      guard !albums.isEmpty else { return }
      DispatchQueue.global(qos: .userInitiated).async {
        let model = ModelTrainer().trainModel(metadata: modelMetadata, on: albums)
        DispatchQueue.main.async {
          self.model = model
        }
      }
    }
    .navigationBarTitle(Text("Training"), displayMode: .inline)
    .toolbar {
      ToolbarItem(placement: .primaryAction) {
        if let model = model {
          Button("Save") {
            completion(model)
          }
          .accessibilityHint("Saves the created model locally.")
        } else {
          Button("Save") {}
            .disabled(true)
            .accessibilityHint(
              "Saves the created model locally. Dimmed until the model has been created.")
        }
      }
    }
  }
}
