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

import SwiftUI
import os

/// Detailed view of data set.
struct DatasetDetailView: View {
  var dataset: Dataset
  let bertQA: BertQAHandler

  @State private var question: String = ""
  @State private var statusMessage: String = "Enter a question."
  @State private var highlightRange: Range<String.Index>? = nil

  @EnvironmentObject var keyboard: KeyboardHeightObserver

  var body: some View {

    GeometryReader { geometry in
      VStack {
        Group {
          ScrollView(.vertical, showsIndicators: true) {
            ContentView(highlightRange: self.highlightRange, content: self.dataset.content)
          }
        }
        .padding(CustomUI.contentViewPadding)
        .onTapGesture { self.hideKeyboard() }

        Spacer()

        VStack(spacing: CustomUI.stackSpacing) {
          StatusView(statusMessage: self.$statusMessage)

          HStack {
            Text("You might want to ask:")
              .font(.footnote)
            Spacer()
          }

          SuggestedQuestionsView(questions: self.dataset.questions, toFill: self.$question)

          HStack {
            TextField(
              "Enter question",
              text: self.$question,
              onEditingChanged: { _ in }
            )
            .textFieldStyle(RoundedBorderTextFieldStyle())
              .animation(.easeOut(duration: CustomUI.keyboardAnimationDuration))

            Button(action: self.tapRunButton) {
              Text("▶︎").font(.title)
            }
            .foregroundColor(Color.orange.opacity(CustomUI.runButtonOpacity))
          }
        }
        .navigationBarTitle(Text(self.dataset.title), displayMode: .inline)
        .padding([.leading, .trailing, .bottom], CGFloat(CustomUI.controlViewPadding))
        .padding(.all, CustomUI.padding)
        .padding(.bottom, max(0, self.keyboard.height - geometry.safeAreaInsets.bottom))
      }
    }
  }

  func tapRunButton() {
    // Clean up previous result.
    highlightRange = nil

    // Hide keyboard.
    self.hideKeyboard()

    // Trim the whitespaces and newlines in the first and end.
    var query = question.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !query.isEmpty else {
      os_log("Textfield failed to filter the empty query.")
      statusMessage = StatusMessage.warnEmptyQuery
      return
    }

    // A query must end with question mark.
    if query.last != "?" {
      query.append("?")
    }

    // Inference the answer with BertQA model.
    guard let result = bertQA.run(query: query, content: dataset.content) else {
      os_log("Failed to inference the answer.")
      statusMessage = StatusMessage.inferenceFailError
      return
    }

    statusMessage = result.description

    // Render the answer in the `contentView`.
    highlightRange = result.answer.text.range
  }

  func hideKeyboard() {
    UIApplication.shared.sendAction(
      #selector(UIResponder.resignFirstResponder),
      to: nil, from: nil, for: nil)
  }
}
