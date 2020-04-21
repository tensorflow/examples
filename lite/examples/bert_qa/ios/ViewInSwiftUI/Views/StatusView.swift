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

/// Status View to inform the result of inference or error.
struct StatusView: View {
  @Binding var statusMessage: String

  var body: some View {
    Text(self.statusMessage)
      .frame(maxWidth: .infinity, minHeight: 40, maxHeight: 40, alignment: .topLeading)
      .padding(.all, CustomUI.padding)
      .font(.system(size: CustomUI.statusFontSize, weight: .semibold))
      .lineLimit(2)
      .background(Color(red: 255 / 255, green: 244 / 255, blue: 229 / 255))
      .cornerRadius(CustomUI.statusTextViewCornerRadius)
  }
}
