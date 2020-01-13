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

import UIKit

class ViewController: UIViewController {
  
private var textClassificationClient: TextClassificationnClient?
  
@IBOutlet weak var scrollView: UIScrollView!
@IBOutlet weak var textView: UITextView!
@IBOutlet weak var textField: UITextField!
  
@IBAction func onPredictTap(_ sender: Any) {
  guard let text = textField.text else { return }
  classify(text)
}
override func viewDidLoad() {
  super.viewDidLoad()
  addKeyboardNotifications()
  DispatchQueue.global().async {
    self.loadClient()
  }
}
  
private func loadClient() {
  textClassificationClient = TextClassificationnClient(modelFileInfo: modelFileInfo, labelsFileInfo: labelsFileInfo, vocabFileInfo: vocabFileInfo)
}
  
} // class ViewController

// Keybaord notifications
extension ViewController {
private func addKeyboardNotifications() {
  NotificationCenter.default.addObserver(self,
                                         selector: #selector(keyboardWillShow(_:)),
                                         name: UIResponder.keyboardWillShowNotification,
                                         object: nil)
  
  NotificationCenter.default.addObserver(self,
                                         selector: #selector(keyboardWillHide(_:)),
                                         name: UIResponder.keyboardWillHideNotification,
                                         object: nil)
}
@objc private func keyboardWillShow(_ notification: Notification) {
  adjustInsetForKeyboardShow(true, notification: notification)
}

@objc private func keyboardWillHide(_ notification: Notification) {
  adjustInsetForKeyboardShow(false, notification: notification)
}
private func adjustInsetForKeyboardShow(_ show: Bool, notification: Notification) {
  guard
    let userInfo = notification.userInfo,
    let keyboardFrame = userInfo[UIResponder.keyboardFrameEndUserInfoKey] as? NSValue else {
    return
  }
  let adjustmentHeight = (keyboardFrame.cgRectValue.height + 20) * (show ? 1 : -1)
  scrollView.contentInset.bottom += adjustmentHeight
  scrollView.verticalScrollIndicatorInsets.bottom += adjustmentHeight
}
} // extension ViewController

extension ViewController {
private func classify(_ text: String) {
  DispatchQueue.global().async {
    guard let results = self.textClassificationClient?.classify(text: text) else { return }
    self.showResults(inputText: text, results: results)
  }
}
private func showResults(inputText: String, results: [Result]) {
  DispatchQueue.main.async {
    var textToShow = "Input: " + inputText + "\nOutput:\n";
    for result in results {
      textToShow.append(result.title + " " + String(result.confidence) + "\n")
    }
    textToShow += "---------\n"
    self.textView.text += textToShow
  }
}
} // extension ViewController
