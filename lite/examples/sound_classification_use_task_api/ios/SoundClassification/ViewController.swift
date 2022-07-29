//
//  ViewController.swift
//  SoundClassification
//
//  Created by MBA0077 on 7/29/22.
//

import UIKit
import AVFoundation
import TensorFlowLiteTaskAudio

class ViewController: UIViewController {
  @IBOutlet weak var tableView: UITableView!

  var datas: [ClassificationCategory] = []

  override func viewDidLoad() {
    super.viewDidLoad()
    startAudioRecognition()
  }

  private func startAudioRecognition() {
    AVAudioSession.sharedInstance().requestRecordPermission { granted in
      if granted {
        self.classification(model: .Yamnet)
      } else {
        self.checkPermissions()
      }
    }
  }

  private func checkPermissions() {
    switch AVAudioSession.sharedInstance().recordPermission {
    case .granted, .undetermined:
      startAudioRecognition()
    case .denied:
      showPermissionsErrorAlert()
    @unknown default:
      fatalError()
    }
  }

  private func classification(model: ModelType) {
    guard let path = Bundle.main.path(forResource: model.fileName, ofType: "tflite") else { return }
    let classifierOptions = AudioClassifierOptions(modelPath: path)
    do {
      let classifier = try AudioClassifier.classifier(options: classifierOptions)
      let inputAudioTensor = classifier.createInputAudioTensor()
      let audioFormat = inputAudioTensor.audioFormat
      let audioTensor = AudioTensor(audioFormat: audioFormat, sampleCount: inputAudioTensor.bufferSize)
      do {
        let audioRecord = try classifier.createAudioRecord()
        func process() {
          do {
            try audioTensor.load(audioRecord: audioRecord)
            let classifier = try classifier.classify(audioTensor: audioTensor)
            let categories = classifier.classifications[0].categories
            let categoriesSorted = categories.sorted {
              $0.score > $1.score
            }
            if categoriesSorted.count > 3 {
              self.datas = Array(categoriesSorted[0..<3])
            } else {
              self.datas = categoriesSorted
            }
            DispatchQueue.main.async {
              self.tableView.reloadData()
            }
          } catch {
            print(error.localizedDescription)
          }
          DispatchQueue.global().asyncAfter(deadline: .now() + 0.2) {
            process()
          }
        }
        try audioRecord.startRecording()
        process()
      } catch { print(error) }
    } catch {
      print(error)
    }
  }
}

extension ViewController {
  private func showPermissionsErrorAlert() {
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

enum ModelType: String {
  case Yamnet = "YAMNet"
  case speechCommandModel = "speech command model"

  var fileName: String {
    switch self {
    case .Yamnet:
      return "lite-model_yamnet"
    case .speechCommandModel:
      return ""
    }
  }
}

extension ViewController: UITableViewDataSource, UITableViewDelegate {

  func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
    guard let cell = tableView.dequeueReusableCell(withIdentifier: "ResultCell") as? ResultTableViewCell else { fatalError() }
    cell.setData(datas[indexPath.row])
    return cell
  }

  func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
    return datas.count
  }
}
