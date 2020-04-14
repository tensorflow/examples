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

class OptionViewController: UIViewController {
  var optionTableView: OptionTableViewController?

  // MARK: Storyboard Connections
  @IBOutlet weak var cancelButton: UIButton!
  @IBOutlet weak var doneButton: UIButton!

  // MARK: - Custom actions
  @IBAction func tapCancelButton() {
    self.dismiss(animated: true, completion: nil)
  }

  @IBAction func tapDoneButton() {
    guard
      let newThreadCount = optionTableView?.threadCountStepper.value
    else {
      os_log("[Option View]: Cannot get the option value", type: .error)
      self.dismiss(animated: true, completion: nil)
      return
    }
    let currentThreadCount = Double(
      UserDefaults.standard.integer(forKey: InterpreterOptions.threadCount.id))

    if currentThreadCount != newThreadCount {
      UserDefaults.standard.set(Int(newThreadCount), forKey: InterpreterOptions.threadCount.id)
    }

    self.dismiss(animated: true, completion: nil)
  }

  // MARK: - Navigation
  override func prepare(for segue: UIStoryboardSegue, sender: Any?) {
    if segue.identifier == "optionTableEmbedingSegue",
      let destination = segue.destination as? OptionTableViewController
    {
      optionTableView = destination
    }
  }
}
