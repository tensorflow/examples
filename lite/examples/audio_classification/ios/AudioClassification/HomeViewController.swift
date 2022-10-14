// Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

import AVFoundation
import TensorFlowLiteTaskAudio
import UIKit

/// The sample app's home screen.
class HomeViewController: UIViewController {

  // MARK: - Variables

  @IBOutlet private weak var tableView: UITableView!
  @IBOutlet private weak var inferenceView: InferenceView!

  private var inferenceResults: [ClassificationCategory] = []

  private var modelType: ModelType = .Yamnet
  private var overLap = 0.5
  private var maxResults = 3
  private var threshold: Float = 0.0
  private var threadCount = 2

  private var audioClassificationHelper: AudioClassificationHelper?

  // MARK: - View controller lifecycle methods
  override func viewDidLoad() {
    super.viewDidLoad()
    inferenceView.setDefault(
      model: modelType, overLab: overLap, maxResult: maxResults, threshold: threshold,
      threads: threadCount)
    inferenceView.delegate = self
    startAudioClassification()
  }

  // MARK: - Private Methods
  /// Request permission and start audio classification if granted.
  private func startAudioClassification() {
    AVAudioSession.sharedInstance().requestRecordPermission { [weak self] granted in
      if granted {
        DispatchQueue.main.async {
          self?.restartClassifier()
        }
      } else {
        self?.checkPermissions()
      }
    }
  }

  /// Check permission and show error if user denied permission.
  private func checkPermissions() {
    switch AVAudioSession.sharedInstance().recordPermission {
    case .granted, .undetermined:
      startAudioClassification()
    case .denied:
      showPermissionsErrorAlert()
    @unknown default:
      fatalError("Microphone permission check returned unexpected result.")
    }
  }

  /// Start a new audio classification routine.
  private func restartClassifier() {
    // Stop the existing classifier if one is running.
    audioClassificationHelper?.stopClassifier()

    // Create a new classifier instance.
    audioClassificationHelper = AudioClassificationHelper(
      modelType: modelType,
      threadCount: threadCount,
      scoreThreshold: threshold,
      maxResults: maxResults)

    // Start the new classification routine.
    audioClassificationHelper?.delegate = self
    audioClassificationHelper?.startClassifier(overlap: overLap)
  }
}

extension HomeViewController {
  private func showPermissionsErrorAlert() {
    let alertController = UIAlertController(
      title: "Microphone Permissions Denied",
      message:
        "Microphone permissions have been denied for this app. You can change this by going to Settings",
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

enum ModelType: String {
  case Yamnet = "YAMNet"
  case speechCommandModel = "Speech Command"

  var fileName: String {
    switch self {
    case .Yamnet:
      return "yamnet"
    case .speechCommandModel:
      return "speech_commands"
    }
  }
}

extension HomeViewController: UITableViewDataSource, UITableViewDelegate {

  func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
    guard
      let cell = tableView.dequeueReusableCell(withIdentifier: "ResultCell") as? ResultTableViewCell
    else { fatalError() }
    cell.setData(inferenceResults[indexPath.row])
    return cell
  }

  func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
    return inferenceResults.count
  }
}

extension HomeViewController: AudioClassificationHelperDelegate {
  func audioClassificationHelper(_ helper: AudioClassificationHelper, didSucceed result: Result) {
    inferenceResults = result.categories
    tableView.reloadData()
    inferenceView.setInferenceTime(result.inferenceTime)
  }

  func audioClassificationHelper(_ helper: AudioClassificationHelper, didFail error: Error) {
    let errorMessage =
      "An error occured while running audio classification: \(error.localizedDescription)"
    let alert = UIAlertController(
      title: "Error", message: errorMessage, preferredStyle: UIAlertController.Style.alert)
    alert.addAction(UIAlertAction(title: "OK", style: UIAlertAction.Style.default, handler: nil))
    present(alert, animated: true, completion: nil)
  }
}

extension HomeViewController: InferenceViewDelegate {
  func view(_ view: InferenceView, needPerformActions action: InferenceView.Action) {
    switch action {
    case .changeModel(let modelType):
      self.modelType = modelType
    case .changeOverlap(let overLap):
      self.overLap = overLap
    case .changeMaxResults(let maxResults):
      self.maxResults = maxResults
    case .changeScoreThreshold(let threshold):
      self.threshold = threshold
    case .changeThreadCount(let threadCount):
      self.threadCount = threadCount
    }

    // Restart the audio classifier as the config as changed.
    restartClassifier()
  }
}
