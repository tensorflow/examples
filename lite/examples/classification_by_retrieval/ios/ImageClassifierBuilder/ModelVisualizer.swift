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

/// Displays the live feed from the camera and overlays classification results.
struct ModelVisualizer: View {
  /// The model to run.
  let model: Model
  /// The queue on which classification is performed.
  private let queue = DispatchQueue(
    label: "com.tensorflow.lite.swift.ClassificationByRetrieval.visualizer")
  /// The classifier running the `model`.
  @State private var classifier: Classifier?
  /// The potential current classification result.
  @State private var result: [Classifier.Label]?
  /// Whether to present the model info UI.
  @State private var showModelInfo = false
  /// The model metadata. This is stored once in order to avoid querying the classifier multiple
  /// times.
  @State private var modelMetadata = ModelMetadata()
  /// The labels from the label map. This is stored once in order to avoid querying the classifier
  /// multiple times.
  @State private var labels: [String] = []
  /// The presentation mode, to programmatically dismiss the visualizer.
  @Environment(\.presentationMode) var presentation

  var body: some View {
    ZStack(alignment: .bottom) {
      ZStack(alignment: .topTrailing) {
        CameraView(queue: queue) { imageBuffer in
          let result = classifier?.classify(pixelBuffer: imageBuffer)
          DispatchQueue.main.async {
            if self.result?.first?.name != result?.first?.name {
              UIAccessibility.post(notification: .announcement, argument: result?.first?.name)
            }
            self.result = result
          }
        }
        .ignoresSafeArea()
        Button {
          showModelInfo = true
        } label: {
          Image(systemName: "info.circle")
            .resizable()
            .frame(width: 24, height: 24)
            .foregroundColor(.primary)
        }
        .buttonStyle(RoundButton())
        .sheet(isPresented: $showModelInfo) {
          ModelInfoView(metadata: modelMetadata, labels: labels)
        }
      }
      if let result = result {
        ResultView(result: result)
          .modifier(BottomPane())
      } else {
        Text("No result")
          .padding()
          .modifier(BottomPane())
      }
    }
    .accessibilityAction(.escape) {
      presentation.wrappedValue.dismiss()
    }
    .navigationBarTitle("", displayMode: .inline)
    .navigationBarHidden(true)
    .onAppear {
      classifier = Classifier(modelURL: model.url)
      if let classifier = classifier {
        labels = classifier.labels()
        modelMetadata = ModelMetadata(
          name: classifier.name, description: classifier.modelDescription,
          author: classifier.author, version: classifier.version, license: classifier.license)
      }
    }
  }

  /// Displays a classification result.
  struct ResultView: View {
    /// The best result, provided that it met a certain threshold.
    private let bestLabels: [Classifier.Label]

    init(result: [Classifier.Label]) {
      bestLabels = result.prefix(Constants.maxLabels).filter { $0.score > Constants.threshold }
    }

    var body: some View {
      VStack {
        ForEach(bestLabels, id: \.self) { label in
          HStack {
            Text(label.name)
            Spacer()
            Text(String(format: "%.2f", label.score))
          }
          ProgressView(value: label.score)
            .progressViewStyle(LinearProgressViewStyle())
        }
      }
      .padding()
    }

    private enum Constants {
      /// The max number of labels to display.
      static let maxLabels = 3
      /// The score threshold for showing labels.
      static let threshold = Float(0.2)
    }
  }

  /// Common styling for views placed as bottom panes.
  struct BottomPane: ViewModifier {
    func body(content: Content) -> some View {
      content
        .frame(maxWidth: .infinity)
        .background(
          VisualEffectView(effect: UIBlurEffect(style: .systemThinMaterial))
            .ignoresSafeArea())
    }
  }

  /// Common styling for buttons.
  struct RoundButton: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
      configuration.label.background(
        VisualEffectView(effect: UIBlurEffect(style: .systemThinMaterial))
          .frame(width: 48, height: 48)
          .clipShape(Circle())
      )
      .offset(x: -24, y: 24)
    }
  }
}
