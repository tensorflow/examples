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
  @IBOutlet weak var inferenceView: InferenceView!

  private var datas: [ClassificationCategory] = []

  private var modelType: ModelType = .Yamnet
  private var overLap: Float = 0.5
  private var maxResults: Int = 3
  private var threshold: Float = 0.3
  private var threadCount: Int = 2

  private var timer: Timer?
  private let processQueue = DispatchQueue(label: "processQueue")

  override func viewDidLoad() {
    super.viewDidLoad()
    inferenceView.setDefault(model: modelType, overLab: overLap, maxResult: maxResults, threshold: threshold, threads: threadCount)
    inferenceView.delegate = self
    startAudioRecognition()
  }

  private func startAudioRecognition() {
    AVAudioSession.sharedInstance().requestRecordPermission { granted in
      if granted {
        self.classification()
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

  private func classification() {
    guard let path = Bundle.main.path(forResource: modelType.fileName, ofType: "tflite") else { return }
    let classifierOptions = AudioClassifierOptions(modelPath: path)
    classifierOptions.baseOptions.computeSettings.cpuSettings.numThreads = threadCount
    classifierOptions.classificationOptions.maxResults = maxResults
    classifierOptions.classificationOptions.scoreThreshold = threshold
    do {
      let classifier = try AudioClassifier.classifier(options: classifierOptions)
      let inputAudioTensor = classifier.createInputAudioTensor()
      let audioFormat = inputAudioTensor.audioFormat
      let audioTensor = AudioTensor(audioFormat: audioFormat, sampleCount: inputAudioTensor.bufferSize)
      let lengthInMilliSeconds = Double(inputAudioTensor.bufferSize) / Double(audioFormat.sampleRate)
      let interval = lengthInMilliSeconds * Double(1 - overLap)
      do {
        let audioRecord = try classifier.createAudioRecord()
        func process() {
          let startTime = Date().timeIntervalSince1970
          do {
            try audioTensor.load(audioRecord: audioRecord)
            let classifier = try classifier.classify(audioTensor: audioTensor)
            let processTime = Date().timeIntervalSince1970 - startTime
            DispatchQueue.main.async {
              self.inferenceView.inferenceTimeLabel.text = "\(Int(processTime * 1000)) ms"
              self.datas = classifier.classifications[0].categories
              self.tableView.reloadData()
            }
          } catch {
            print(error.localizedDescription)
          }
        }
        try audioRecord.startRecording()
        timer?.invalidate()
        print(interval)
        timer = Timer.scheduledTimer(withTimeInterval: interval, repeats: true, block: { _ in
          self.processQueue.async {
            process()
          }
        })
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

extension ViewController: InferenceViewDelegate {
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
    classification()
  }
}
