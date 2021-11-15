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
import ios_ImageClassifierBuilderLibObjC

/// Displays the list of models available to the app.
struct ModelList: View {
  /// The manager of the list of models.
  @StateObject private var modelData = ModelData()
  /// Whether the model creation UI is currently presented.
  @State private var showModelCreationUI = false
  /// Whether the model sharing UI is currently presented.
  @State private var showModelSharingUI = false
  /// The error message to display in an alert.
  @State private var errorAlertMessage: String?
  /// Whether the delete confirmation sheet it is currently presented.
  @State private var showDeleteConfirmation = false

  var body: some View {
    NavigationView {
      VStack {
        if modelData.models.isEmpty {
          Text("To create your first model, tap the \"+\" icon in the toolbar.")
            .padding()
        }
        List {
          Section(header: Text("Models")) {
            ForEach(modelData.models.indices, id: \.self) { index in
              let model = modelData.models[index]
              NavigationLink(destination: ModelVisualizer(model: model)) {
                Row(model: model)
                  .contextMenu {
                    makeShareButton()
                    makeDeleteButton(index: index)
                  }
                  .sheet(isPresented: $showModelSharingUI) {
                    ShareSheet(modelURL: model.url) {
                      showModelSharingUI = false
                    }
                  }
              }
              .swipeActions {
                makeDeleteButton(index: index)
                makeShareButton()
              }
              .confirmationDialog(
                "Are you sure you want to delete \"\(model.name)\"?",
                isPresented: $showDeleteConfirmation,
                titleVisibility: .visible
              ) {
                Button("Delete", role: .destructive) { removeModel(at: index) }
              }
            }
            .accessibilityAction(named: Text("Share")) {
              showModelSharingUI.toggle()
            }
          }
        }
        .listStyle(GroupedListStyle())
      }
      .navigationBarTitle(Text("Image Classifier Builder"), displayMode: .inline)
      .alert(item: $errorAlertMessage) { message in
        Alert(title: Text("Error"), message: Text(message), dismissButton: .default(Text("OK")))
      }
      .sheet(isPresented: $showModelCreationUI) {
        NavigationView {
          ModelCreationView { model in
            add(model)
            showModelCreationUI = false
          }
        }
      }
      .toolbar {
        ToolbarItem(placement: .primaryAction) {
          Button {
            showModelCreationUI.toggle()
          } label: {
            Image(systemName: "plus")
          }
          .accessibilityLabel("Create a model")
          .accessibilityHint("Opens the model creation flow.")
        }
        ToolbarItem(placement: .navigationBarLeading) {
          Link(
            destination: URL(
              string:
                "https://github.com/tensorflow/examples/blob/master/lite/examples/classification_by_retrieval/README.md"
            )!
          ) {
            Image(systemName: "questionmark.circle")
              .accessibilityLabel("Help")
              .accessibilityHint("Opens the help page.")
          }
        }
      }
    }
    .navigationViewStyle(.stack)
    .onAppear {
      loadModelData()
    }
    .onOpenURL { url in
      do {
        try importModel(at: url)
      } catch {
        errorAlertMessage = error.localizedDescription
      }
    }
  }

  /// Loads the models from disk.
  private func loadModelData() {
    do {
      try modelData.load()
    } catch {
      fatalError("Failed to load the model data: \(error)")
    }
  }

  /// Saves the models to disk.
  private func saveModelData() {
    do {
      try modelData.save()
    } catch {
      fatalError("Failed to save the model data: \(error)")
    }
  }

  /// Adds a model to the list.
  ///
  /// It is inserted in the first position, then the model data is saved immediately.
  private func add(_ model: Model?) {
    if let model = model {
      modelData.models.insert(model, at: 0)
      saveModelData()
    }
  }

  /// Removes a model from the list.
  ///
  /// The model data is saved immediately.
  private func removeModel(at index: Int) {
    let model = modelData.models[index]
    do {
      try FileManager.default.removeItem(at: model.url)
    } catch {
      errorAlertMessage = "Couldn't delete \(model.name): \(error.localizedDescription)"
      return
    }
    modelData.models.remove(at: index)
    saveModelData()
  }

  /// Imports a model from a file URL.
  private func importModel(at url: URL) throws {
    guard url.pathExtension == "tflite" else {
      throw RuntimeError("File extension is not `tflite`.")
    }

    // Compose the new URL on disk.
    let name = url.deletingPathExtension().lastPathComponent
    let modelURL = Model.makeURLForNewModel(named: name)

    // Copy the file to the location within the app.
    guard url.startAccessingSecurityScopedResource() else {
      throw RuntimeError("Couldn't access the security-scoped file at \(url)")
    }
    defer {
      url.stopAccessingSecurityScopedResource()
    }
    do {
      try FileManager.default.copyItem(at: url, to: modelURL)
    } catch {
      throw RuntimeError(
        "Couldn't make a local copy of the security-scoped file at \(url): \(error)")
    }

    // Add a new Model instance.
    let labelsCount = Classifier(modelURL: modelURL).labels().count
    let size = (try? url.resourceValues(forKeys: [.fileSizeKey]))?.fileSize.map(Int64.init)
    add(Model(name: name, labelsCount: labelsCount, size: size))
  }

  private func makeShareButton() -> some View {
    Button {
      showModelSharingUI.toggle()
    } label: {
      Label("Share", systemImage: "square.and.arrow.up")
    }
  }

  private func makeDeleteButton(index: Int) -> some View {
    Button(role: .destructive) {
      showDeleteConfirmation = true
    } label: {
      Label("Delete", systemImage: "trash")
    }
  }

  /// Displays a model in the list.
  struct Row: View {
    /// The represented model.
    let model: Model
    /// The model size formatter.
    static private var formatter: ByteCountFormatter = {
      let formatter = ByteCountFormatter()
      formatter.allowedUnits = [.useMB]
      formatter.countStyle = .memory
      return formatter
    }()

    var body: some View {
      HStack {
        HStack(alignment: .lastTextBaseline) {
          Text(model.name)
          Text("\(model.labelsCount) labels")
            .font(.caption)
            .foregroundColor(.secondary)
          if let modelSize = model.size {
            Text(Row.formatter.string(fromByteCount: modelSize))
              .font(.caption)
              .foregroundColor(.secondary)
          }
        }
      }
    }
  }
}
