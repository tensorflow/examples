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
import os

/// Controller for detailed view of each `Dataset`.
class DatasetDetailViewController: UIViewController {
  /// Selected `Dataset` to run the Bert model.
  var dataset: Dataset?

  var bertQA: BertQAHandler?

  // MARK: Views created at Storyboard
  @IBOutlet weak var contentView: UITextView!

  @IBOutlet weak var statusTextView: UITextView!
  @IBOutlet weak var questionField: UITextField!
  @IBOutlet weak var runButton: UIButton!

  // MARK: View created programmatically
  @IBOutlet weak var suggestedQuestionsStackView: UIStackView!

  // MARK: Custom init and deinit
  required init?(coder: NSCoder) {
    super.init(coder: coder)

    // Add obserber for keyboard notification to adjust view size.
    NotificationCenter.default.addObserver(
      self, selector: #selector(keyboardWillShow(notification:)),
      name: UIResponder.keyboardWillShowNotification, object: nil
    )
    NotificationCenter.default.addObserver(
      self, selector: #selector(keyboardWillHide(notification:)),
      name: UIResponder.keyboardWillHideNotification, object: nil
    )
  }

  deinit {
    // Remove added observer.
    NotificationCenter.default.removeObserver(self)
  }

  // MARK: - View handling methods
  override func viewDidLoad() {
    super.viewDidLoad()
    guard let dataset = dataset else {
      fatalError("Data set was not passed correctly.")
    }

    navigationItem.title = dataset.title

    // Make status text view to have rounded border.
    statusTextView.layer.cornerRadius = CustomUI.statusTextViewCornerRadius

    // Add content of the data set to the view.
    contentView.text = dataset.content
    contentView.font = .systemFont(ofSize: 17)

    // Add suggested questions as buttons to the stack view.
    for question in dataset.questions {
      let button = SuggestedQuestionButton(of: question)
      suggestedQuestionsStackView.addArrangedSubview(button)
      button.addTarget(
        self,
        action: #selector(tapSuggestedQuestionButton(of:)),
        for: .touchUpInside)
    }

    // Disable run button at the begining as it is empty.
    runButton.isEnabled = false
  }

  // MARK: Action for keyboard notification
  @objc func keyboardWillShow(notification: Notification) {
    // Get keyboard height.
    let keyboardHeight: CGFloat
    if let keyboardFrame =
      (notification.userInfo?[UIResponder.keyboardFrameEndUserInfoKey] as? NSValue)?
      .cgRectValue
    {
      keyboardHeight = keyboardFrame.height
    } else {
      keyboardHeight = 0
    }

    // Resize the view adjusting to the keyboard height.
    view.frame.size.height = UIScreen.main.bounds.height - keyboardHeight
  }

  @objc func keyboardWillHide(notification: Notification) {
    // Resize the view to fil the screen.
    view.frame.size.height = UIScreen.main.bounds.height
  }

  // MARK: - Custom actions
  @IBAction func textFieldDidChange(textField: UITextField) {
    // Drop empty question.
    guard let query = textField.text?.trimmingCharacters(in: .whitespacesAndNewlines),
      !query.isEmpty
    else {
      os_log("Got empty query.")
      statusTextView.text = StatusMessage.warnEmptyQuery
      runButton.isEnabled = false
      return
    }

    // Ask to run if query is not empty.
    statusTextView.text = StatusMessage.askRun
    runButton.isEnabled = true
  }

  @IBAction func tapContentView() {
    // Hide keyboard.
    questionField.resignFirstResponder()
  }

  @IBAction func tapSuggestedQuestionButton(of sender: SuggestedQuestionButton!) {
    // Fill the `questionField` in the view with selected question.
    questionField.text = sender.title(for: .normal)

    // Activate the button and ask to run.
    statusTextView.text = StatusMessage.askRun
    runButton.isEnabled = true
  }

  @IBAction func tapRunButton() {
    // Disable run button until getting the answer.
    runButton.isEnabled = false

    // Clean up previous result.
    contentView.textStorage
      .removeAttribute(
        .backgroundColor,
        range: NSRange(location: 0, length: contentView.text.count))

    // Hide keyboard.
    questionField.resignFirstResponder()

    // Trim the whitespaces and newlines in the first and end.
    guard var query = questionField.text?.trimmingCharacters(in: .whitespacesAndNewlines),
      !query.isEmpty
    else {
      os_log("Textfield failed to filter the empty query.")
      statusTextView.text = StatusMessage.warnEmptyQuery
      return
    }

    // A query must end with question mark.
    if query.last != "?" {
      query.append("?")
    }

    // Inference the answer with BertQA model.
    guard let result = bertQA?.run(query: query, content: contentView.text) else {
      os_log("Failed to inference the answer.")
      statusTextView.text = StatusMessage.inferenceFailError
      return
    }

    statusTextView.text = result.description

    // Render the answer in the `contentView`.
    contentView.textStorage
      .addAttribute(
        .backgroundColor,
        value: CustomUI.textHighlightColor,
        range: NSRange(result.answer.text.range, in: contentView.text)
      )

    // Enable button as the process is finished.
    runButton.isEnabled = true
  }
}
