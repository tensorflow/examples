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

class OptionTableViewController: UITableViewController {
  // MARK: Storyboard Connections
  @IBOutlet weak var threadCountLabel: UILabel!
  @IBOutlet weak var threadCountStepper: UIStepper!

  // MARK: - View handling methods
  override func viewDidLoad() {
    // Initialize UI properties with user default value.
    let threadCount = UserDefaults.standard.integer(forKey: InterpreterOptions.threadCount.id)
    threadCountLabel.text = threadCount.description
    threadCountStepper.value = Double(threadCount)

    // Set thread count limits.
    threadCountStepper.maximumValue = Double(InterpreterOptions.threadCount.maximumValue)
    threadCountStepper.minimumValue = Double(InterpreterOptions.threadCount.minimumValue)
  }

  // MARK: Button Actions
  @IBAction func didChangeThreadCount(_ sender: UIStepper) {
    threadCountLabel.text = Int(sender.value).description
  }

  @IBAction func didResetOptions() {
    threadCountLabel.text = InterpreterOptions.threadCount.defaultValue.description
    threadCountStepper.value = Double(InterpreterOptions.threadCount.defaultValue)
  }
}
