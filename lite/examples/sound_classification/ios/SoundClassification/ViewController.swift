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

import UIKit

class ViewController: UIViewController {
  // MARK: - Variables
  @IBOutlet weak var tableView: UITableView!

  private var soundClassifier: SoundClassifier!
  private var categories: [ClassificationCategory] = []

  // MARK: - View controller lifecycle methods

  override func viewDidLoad() {
    super.viewDidLoad()


    tableView.dataSource = self
    tableView.backgroundColor = .white
    tableView.tableFooterView = UIView()

    soundClassifier = SoundClassifier(modelFileName: "yamnet_audio_classifier_with_metadata", delegate: self)
    soundClassifier.startAudioClassificationOnMicInput()
  }

  // MARK: - Private Methods
}


extension ViewController {
  func audioInputManagerDidFailToAchievePermission(_ audioInputManager: AudioInputManager) {
    let alertController = UIAlertController(
      title: "Microphone Permissions Denied",
      message: "Microphone permissions have been denied for this app. You can change this by going to Settings",
      preferredStyle: .alert
    )

    let cancelAction = UIAlertAction(title: "Cancel", style: .cancel, handler: nil)
    let settingsAction = UIAlertAction(title: "Settings", style: .default) { _ in
      UIApplication.shared.open(
        URL(string: UIApplication.openSettingsURLString)!,
        options: [:],
        completionHandler: nil
      )
    }
    alertController.addAction(cancelAction)
    alertController.addAction(settingsAction)

    present(alertController, animated: true, completion: nil)
  }
}

extension ViewController: SoundClassifierDelegate {
  
  func soundClassifier(_ soundClassifier: SoundClassifier, didClassifyWithCategories categories: [ClassificationCategory]) {
    guard let label = categories[0].label else {
      return
    }
    print("Label: \(label), Score: \(categories[0].score)");
  }
}

// MARK: - UITableViewDataSource
extension ViewController: UITableViewDataSource {
  func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
    return categories.count
  }

  func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
    guard let cell = tableView.dequeueReusableCell(
      withIdentifier: "probabilityCell",
      for: indexPath
    ) as? ProbabilityTableViewCell else { return UITableViewCell() }

//    cell.label.text = soundClassifier.labelNames[indexPath.row]
    UIView.animate(withDuration: 0.4) {
      cell.progressView.setProgress(self.categories[indexPath.row].score, animated: true)
    }
    return cell
  }
}
