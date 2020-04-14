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

/// Observes keyboard height events.
final class KeyboardHeightObserver: ObservableObject {
  @Published private(set) var height: CGFloat = 0

  init() {
    NotificationCenter.default.addObserver(
      self, selector: #selector(keyBoardWillShow(notification:)),
      name: UIResponder.keyboardWillShowNotification, object: nil)
    NotificationCenter.default.addObserver(
      self, selector: #selector(keyBoardWillHide(notification:)),
      name: UIResponder.keyboardWillHideNotification, object: nil)
  }

  deinit {
    NotificationCenter.default.removeObserver(self)
  }

  @objc func keyBoardWillShow(notification: Notification) {
    if let keyboardFrame =
      (notification.userInfo?[UIResponder.keyboardFrameEndUserInfoKey] as? NSValue)?
      .cgRectValue
    {
      height = keyboardFrame.height
    } else {
      height = 0
    }
  }

  @objc func keyBoardWillHide(notification: Notification) {
    height = 0
  }
}
