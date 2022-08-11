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

import TensorFlowLiteTaskAudio

protocol AudioClassificationHelperDelegate {
  func sendResult(_ result: Result)
}

/// Stores results for a particular frame that was successfully run through the `Interpreter`.
struct Result {
  let inferenceTime: Double
  let categories: [ClassificationCategory]
}

/// Information about a model file or labels file.
typealias FileInfo = (name: String, extension: String)

/// This class handles all data preprocessing and makes calls to run inference on a record
/// by invoking the `AudioClassifier`.
class AudioClassificationHelper {

  // MARK: Public properties
  var delegate: AudioClassificationHelperDelegate?

  // MARK: Private properties
  /// TensorFlow Lite `AudioClassifier` object for performing object detection using a given model.
  private var classifier: AudioClassifier
  private var audioRecord: AudioRecord
  private var timer: Timer?
  private let processQueue = DispatchQueue(label: "processQueue")

  // MARK: - Initialization

  /// A failable initializer for `SoundClassificationHelper`. A new instance is created if the model and
  /// labels files are successfully loaded from the app's main bundle.
  init?(modelType: ModelType, threadCount: Int, scoreThreshold: Float, maxResults: Int) {
    let modelFilename = modelType.fileName

    // Construct the path to the model file.
    guard let modelPath = Bundle.main.path(
      forResource: modelFilename,
      ofType: "tflite"
    ) else {
      print("Failed to load the model file with name: \(modelFilename).")
      return nil
    }

    // Specify the options for the `Classifier`.
    let classifierOptions = AudioClassifierOptions(modelPath: modelPath)
    classifierOptions.baseOptions.computeSettings.cpuSettings.numThreads = threadCount
    classifierOptions.classificationOptions.maxResults = maxResults
    classifierOptions.classificationOptions.scoreThreshold = scoreThreshold
    do {
      // Create the `Classifier`.
      classifier = try AudioClassifier.classifier(options: classifierOptions)
      audioRecord = try classifier.createAudioRecord()
    } catch let error {
      print("Failed to create the interpreter with error: \(error.localizedDescription)")
      return nil
    }
  }

  deinit {
    audioRecord.stop()
  }

  /// This class handles all data preprocessing and delegate results to Controller when classtifi is done
  func runClassifier(overLap: Double) {
    let inputAudioTensor = classifier.createInputAudioTensor()
    let audioFormat = inputAudioTensor.audioFormat
    let audioTensor = AudioTensor(audioFormat: audioFormat, sampleCount: inputAudioTensor.bufferSize)
    do {
      func processing() {
        let startTime = Date().timeIntervalSince1970
        do {
          try audioTensor.load(audioRecord: audioRecord)
          let results = try classifier.classify(audioTensor: audioTensor)
          let inferenceTime = Date().timeIntervalSince1970 - startTime
          DispatchQueue.main.async {
            // Send datas to Controller
            self.delegate?.sendResult(
              Result(inferenceTime: inferenceTime,
                     categories: results.classifications[0].categories)
            )
          }
        } catch {
          print(error.localizedDescription)
        }
      }
      // Start recording audio
      try audioRecord.startRecording()
      // Calculate interval between sampling based on overlap
      let lengthInMilliSeconds = Double(inputAudioTensor.bufferSize) / Double(audioFormat.sampleRate)
      let interval = lengthInMilliSeconds * Double(1 - overLap)
      timer?.invalidate()
      // Run the process after every fixed interval
      timer = Timer.scheduledTimer(withTimeInterval: interval, repeats: true, block: { [weak self] _ in
        self?.processQueue.async {
          processing()
        }
      })
    } catch { print(error.localizedDescription) }
  }
}

