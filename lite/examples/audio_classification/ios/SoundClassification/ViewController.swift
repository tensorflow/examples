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

  private var audioInputManager: AudioInputManager!
  private var soundClassifier: SoundClassifier!
  private var bufferSize: Int = 0
  private var probabilities: [Float32] = []

  // MARK: - View controller lifecycle methods

  override func viewDidLoad() {
    super.viewDidLoad()

    tableView.dataSource = self
    tableView.backgroundColor = .white
    tableView.tableFooterView = UIView()

    soundClassifier = SoundClassifier(modelFileName: "sound_classification", delegate: self)

    startAudioRecognition()
  }

  // MARK: - Private Methods

  /// Initializes the AudioInputManager and starts recognizing on the output buffers.
  private func startAudioRecognition() {
    audioInputManager = AudioInputManager(sampleRate: soundClassifier.sampleRate)
    audioInputManager.delegate = self

    bufferSize = audioInputManager.bufferSize

    audioInputManager.checkPermissionsAndStartTappingMicrophone()
  }

  private func runModel(inputBuffer: [Int16]) {
    soundClassifier.start(inputBuffer: inputBuffer)
  }
}

extension ViewController: AudioInputManagerDelegate {
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

  func audioInputManager(
    _ audioInputManager: AudioInputManager,
    didCaptureChannelData channelData: [Int16]
  ) {
    let sampleRate = soundClassifier.sampleRate
    self.runModel(inputBuffer: Array(channelData[0..<sampleRate]))
    self.runModel(inputBuffer: Array(channelData[sampleRate..<bufferSize]))
  }
}

extension ViewController: SoundClassifierDelegate {
  func soundClassifier(
    _ soundClassifier: SoundClassifier,
    didInterpreteProbabilities probabilities: [Float32]
  ) {
    self.probabilities = probabilities
    DispatchQueue.main.async {
      self.tableView.reloadData()
    }
  }
}

// MARK: - UITableViewDataSource
extension ViewController: UITableViewDataSource {
  func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
    return probabilities.count
  }

  func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
    guard let cell = tableView.dequeueReusableCell(
      withIdentifier: "probabilityCell",
      for: indexPath
    ) as? ProbabilityTableViewCell else { return UITableViewCell() }

    cell.label.text = soundClassifier.labelNames[indexPath.row]
    UIView.animate(withDuration: 0.4) {
      cell.progressView.setProgress(self.probabilities[indexPath.row], animated: true)
    }
    return cell
  }
}
