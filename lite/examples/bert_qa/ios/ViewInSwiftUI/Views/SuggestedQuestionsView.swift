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

/// A container view of a list of suggested question buttons.
struct SuggestedQuestionsView: View {
  var questions: [String]

  @Binding var toFill: String

  var body: some View {
    ScrollView(.horizontal, showsIndicators: false) {
      HStack {
        ForEach(questions, id: \.self) { question in
          Button(action: { self.toFill = question }) {
            SuggestedQuestionText(text: question)
          }
          .buttonStyle(SuggestedQuestionStyle())
        }
      }
    }
  }
}

/// Text on a suggested question button.
struct SuggestedQuestionText: View {
  var text: String

  var body: some View {
    Text(text)
      .font(.caption)
      .fontWeight(.medium)
      .foregroundColor(.black)
      .padding(CustomUI.textPadding)
      .padding([.leading, .trailing], CustomUI.textSidePadding)
  }
}

/// Style of a suggested question button.
struct SuggestedQuestionStyle: ButtonStyle {
  func makeBody(configuration: Self.Configuration) -> some View {
    configuration.label.background(
      RoundedRectangle(cornerRadius: CustomUI.suggestedQuestionCornerRadius)
        .stroke(
          LinearGradient(
            gradient: Gradient(colors: [Color.orange, Color.orange.opacity(0.7), Color.yellow]),
            startPoint: .topLeading, endPoint: .bottomTrailing),
          lineWidth: 1
        )
        .padding(1)
    ).scaleEffect(configuration.isPressed ? 0.95 : 1.0)
  }
}
