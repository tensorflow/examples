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

import TensorFlowLiteTaskAudio

/// Delegate to returns the classification results.
protocol AudioClassificationHelperDelegate: AnyObject {
  func audioClassificationHelper(_ helper: AudioClassificationHelper, didSucceed result: Result)
  func audioClassificationHelper(_ helper: AudioClassificationHelper, didFail error: Error)
}

private let errorDomain = "org.tensorflow.lite.examples"

/// Stores results for a particular audio snipprt that was successfully classified.
struct Result {
  let inferenceTime: Double
  let categories: [ClassificationCategory]
}

/// Information about a model file.
typealias FileInfo = (name: String, extension: String)

/// This class handles all data preprocessing and makes calls to run inference on a audio snippet
/// by invoking the Task Library's `AudioClassifier`.
class AudioClassificationHelper {

  // MARK: Public properties

  weak var delegate: AudioClassificationHelperDelegate?

  // MARK: Private properties

  /// An `AudioClassifier` object for performing audio classification using a given model.
  private let classifier: AudioClassifier

  /// An object to continously record audio using the device's microphone.
  private let audioRecord: AudioRecord

  /// A tensor to store the input audio for the model.
  private let inputAudioTensor: AudioTensor

  /// A timer to schedule classification routine to run periodically.
  private var timer: Timer?

  /// A queue to offload the classification routine to a background thread.
  private let processQueue = DispatchQueue(label: "processQueue")

  // MARK: - Initialization

  /// A failable initializer for `AudioClassificationHelper`.
  ///
  /// A new instance is created if the model is successfully loaded from the app's main bundle.
  init?(modelType: ModelType, threadCount: Int, scoreThreshold: Float, maxResults: Int) {
    let modelFilename = modelType.fileName

    // Construct the path to the model file.
    guard
      let modelPath = Bundle.main.path(
        forResource: modelFilename,
        ofType: "tflite"
      )
    else {
      print("Failed to load the model file \(modelFilename).tflite.")
      return nil
    }

    // Specify the options for the classifier.
    let classifierOptions = AudioClassifierOptions(modelPath: modelPath)
    classifierOptions.baseOptions.computeSettings.cpuSettings.numThreads = threadCount
    classifierOptions.classificationOptions.maxResults = maxResults
    classifierOptions.classificationOptions.scoreThreshold = scoreThreshold

    do {
      // Create the classifier.
      classifier = try AudioClassifier.classifier(options: classifierOptions)

      // Create an `AudioRecord` instance to record input audio that satisfies
      // the model's requirements.
      audioRecord = try classifier.createAudioRecord()
      inputAudioTensor = classifier.createInputAudioTensor()
    } catch {
      print("Failed to create the classifier with error: \(error.localizedDescription)")
      return nil
    }
  }

  func stopClassifier() {
    audioRecord.stop()
    timer?.invalidate()
    timer = nil
  }

  /// Start the audio classification routine in the background.
  ///
  /// Classification results are periodically returned to the delegate.
  /// - Parameters overlap: Overlapping factor between consecutive audio snippet to be classified.
  ///   Value must be >= 0 and < 1.
  func startClassifier(overlap: Double) {
    if overlap < 0 {
      let error = NSError(
        domain: errorDomain,
        code: 0,
        userInfo: [NSLocalizedDescriptionKey: "overlap must be equal or larger than 0."])
      delegate?.audioClassificationHelper(self, didFail: error)
    }

    if overlap >= 1 {
      let error = NSError(
        domain: errorDomain, code: 0,
        userInfo: [NSLocalizedDescriptionKey: "overlap must be smaller than 1."])
      delegate?.audioClassificationHelper(self, didFail: error)
    }

    do {
      // Start recording audio.
      try audioRecord.startRecording()

      // Calculate interval between sampling based on overlap.
      let audioFormat = inputAudioTensor.audioFormat
      let lengthInMilliSeconds =
        Double(inputAudioTensor.bufferSize) / Double(audioFormat.sampleRate)
      let interval = lengthInMilliSeconds * Double(1 - overlap)
      timer?.invalidate()

      // Schedule the classification routine to run every fixed interval.
      timer = Timer.scheduledTimer(withTimeInterval: interval, repeats: true) {
        [weak self] _ in
        self?.processQueue.async {
          self?.runClassification()
        }
      }
    } catch {
      delegate?.audioClassificationHelper(self, didFail: error)
    }
  }

  /// Run the classification routine with the latest audio stored in the `AudioRecord` instance's
  /// buffer.
  private func runClassification() {
    let startTime = Date().timeIntervalSince1970
    do {
      // Grab the latest audio chunk in the audio record and run classification.
      try inputAudioTensor.load(audioRecord: audioRecord)
      let results = try classifier.classify(audioTensor: inputAudioTensor)
      let inferenceTime = Date().timeIntervalSince1970 - startTime

      // Return the classification result to the delegate.
      DispatchQueue.main.async {
        let result = Result(
          inferenceTime: inferenceTime,
          categories: results.classifications[0].categories)
        // Send classification result to the delegate.
        self.delegate?.audioClassificationHelper(self, didSucceed: result)
      }
    } catch {
      delegate?.audioClassificationHelper(self, didFail: error)
    }
  }
}
