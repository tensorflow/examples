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
import UIKit

/// Adds support for Share (UIActivityViewController) in Swift UI.
struct ShareSheet: UIViewControllerRepresentable {
  /// The model URL to share.
  let modelURL: URL
  /// The closure to call once the Share sheet did finish.
  let completion: () -> Void

  func makeUIViewController(context: Context) -> UIActivityViewController {
    let activityViewController =
      UIActivityViewController(activityItems: [modelURL], applicationActivities: nil)
    activityViewController.completionWithItemsHandler = {
      (_: UIActivity.ActivityType?, _: Bool, _: [Any]?, _: Error?) in
      completion()
    }
    return activityViewController
  }

  func updateUIViewController(_ uiViewController: UIActivityViewController, context: Context) {
  }
}
