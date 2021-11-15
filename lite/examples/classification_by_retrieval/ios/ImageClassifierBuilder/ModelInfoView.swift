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

/// Displays the info located in the metadata of a given model.
struct ModelInfoView: View {
  @Environment(\.presentationMode) var presentationMode

  /// The model metadata.
  let metadata: ModelMetadata

  /// The labels from the label map.
  let labels: [String]

  var body: some View {
    NavigationView {
      Form {
        KeyValueRow(key: "Name", value: metadata.name)
        KeyValueRow(key: "Description", value: metadata.description)
        KeyValueRow(key: "Author", value: metadata.author)
        KeyValueRow(key: "Version", value: metadata.version)
        KeyValueRow(key: "License", value: metadata.license)
        Section(header: Text("Labels")) {
          ForEach(labels) { label in
            Text(label)
          }
        }
      }
      .navigationBarTitle(Text("Model Info"), displayMode: .inline)
      .toolbar {
        ToolbarItem(placement: .confirmationAction) {
          Button("OK") {
            presentationMode.wrappedValue.dismiss()
          }
        }
      }
    }
  }

  struct KeyValueRow: View {
    let key: String
    let value: String

    var body: some View {
      HStack {
        Text(key)
        Spacer()
        Text(value)
      }
      .accessibilityElement(children: .combine)
    }
  }
}
