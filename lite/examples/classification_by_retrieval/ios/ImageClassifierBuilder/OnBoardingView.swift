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

struct OnBoardingView: View {
  /// The closure to call once the Get Started button is tapped.
  let completion: () -> Void

  var body: some View {
    VStack {
      Text("Image Classifier Builder")
        .font(.largeTitle)
        .padding()
      Text(
        """
        This app is designed for easy object classifier prototyping, not for any application \
        that requires a high level of accuracy. It is not designed and will not work well for \
        person classification. Please visit our \
        [source code page](https://github.com/tensorflow/examples/blob/master/lite/examples/classification_by_retrieval/README.md) \
        for technical details and responsible model building.
        """
      )
      .multilineTextAlignment(.center)
      .padding()
      Button {
        completion()
      } label: {
        Text("Get Started")
      }
      .buttonStyle(.borderedProminent)
      .padding()
    }
  }
}

@available(iOS 13.0, *)
struct OnBoardingView_Previews: PreviewProvider {
  static var previews: some View {
    OnBoardingView {}
  }
}
