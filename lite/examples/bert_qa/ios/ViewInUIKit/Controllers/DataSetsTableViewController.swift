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

/// Controller for a table for the list of `Dataset`s.
class DatasetsTableViewController: UITableViewController {
  /// List of `Dataset`s to run the Bert model.
  var datasets: [Dataset] = Dataset.load()
  var bertQA: BertQAHandler?

  required init?(coder: NSCoder) {
    super.init(coder: coder)

    // Add obserber for interpreter option changing.
    NotificationCenter.default.addObserver(
      self, selector: #selector(interpreterOptionDidChange(notification:)),
      name: UserDefaults.didChangeNotification, object: nil
    )
  }

  deinit {
    // Remove added observer.
    NotificationCenter.default.removeObserver(self)
  }

  // MARK: - View handling methods
  override func viewDidLoad() {
    super.viewDidLoad()

    updateBertQA()
  }

  // MARK: - Table view data source

  override func numberOfSections(in tableView: UITableView) -> Int {
    return 1
  }

  override func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
    return datasets.count
  }

  override func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath)
    -> UITableViewCell
  {
    let cell =
      tableView.dequeueReusableCell(withIdentifier: "DatasetTitleCell") as! DatasetTitleCell
    cell.titleLabel.text = datasets[indexPath.row].title

    return cell
  }

  // MARK: - Navigation

  override func prepare(for segue: UIStoryboardSegue, sender: Any?) {
    // Hand over `Dataset` of the selected title to the `DatasetDetailViewController`.
    guard
      segue.identifier == "ChooseDataset",
      let datasetDetailViewController = segue.destination as? DatasetDetailViewController,
      let selectedIndex = tableView.indexPathForSelectedRow?.row
    else {
      return
    }
    datasetDetailViewController.dataset = datasets[selectedIndex]
    datasetDetailViewController.bertQA = bertQA
  }

  // MARK: Update BertQA on changing interpreter option
  @objc func interpreterOptionDidChange(notification: Notification) {
    updateBertQA()
  }

  private func updateBertQA() {
    let threadCount = UserDefaults.standard.integer(forKey: InterpreterOptions.threadCount.id)

    do {
      bertQA = try BertQAHandler(threadCount: threadCount)
    } catch let error {
      fatalError(error.localizedDescription)
    }
  }
}
