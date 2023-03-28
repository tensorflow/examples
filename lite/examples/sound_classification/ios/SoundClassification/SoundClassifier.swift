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

import Foundation

public protocol SoundClassifierDelegate: class {
  func soundClassifier(
    _ soundClassifier: SoundClassifier,
    didClassifyWithCategories categories: [ClassificationCategory]
  )
}

/// Performs classification on sound.
/// The API supports models which accept sound input via `Int16` sound buffer and one classification output tensor.
/// The output of the recognition is emitted as delegation.
public class SoundClassifier {
  // MARK: - Constants
  private let modelFileName: String
  private let modelFileExtension: String
  private let labelFilename: String
  private let labelFileExtension: String
  private let audioBufferInputTensorIndex: Int = 0
  
  private var audioRecord: AudioRecord?
  private var audioTensor: AudioTensor?
  private var audioClassifier: AudioClassifier?

  // MARK: - Variables
  public weak var delegate: SoundClassifierDelegate?

  /// Sample rate for input sound buffer. Caution: generally this value is a bit less than 1 second's audio sample.
  private(set) var sampleRate = 0
  /// Lable names described in the lable file
//  private(set) var labelNames: [String] = []
//  private var interpreter: Interpreter!
  private var timer: Timer?

  // MARK: - Public Methods

  public init(
    modelFileName: String,
    modelFileExtension: String = "tflite",
    labelFilename: String = "labels",
    labelFileExtension: String = "txt",
    delegate: SoundClassifierDelegate? = nil
  ) {
    self.modelFileName = modelFileName
    self.modelFileExtension = modelFileExtension
    self.labelFilename = labelFilename
    self.labelFileExtension = labelFileExtension
    self.delegate = delegate
    
    
    guard let modelPath = Bundle.main.path(
      forResource: modelFileName,
      ofType: modelFileExtension
    ) else { return }
    
    do {
      self.audioClassifier = try AudioClassifier(modelPath:modelPath);
      self.audioRecord = try audioClassifier?.createAudioRecord();
      self.audioTensor = try audioClassifier?.createInputAudioTensor()
    } catch {
      print("Failed to set up the audio classifier with error: \(error.localizedDescription)")
    }
  }
  
  public func startAudioClassificationOnMicInput() {
  
      AVAudioSession.sharedInstance().requestRecordPermission {[weak self] granted in
        do {
            try self?.audioRecord?.startRecording()
            self?.startAudioClassification()
          }
        catch {
          print(error.localizedDescription)
        }
      }
    }
  
  // MARK: - Private Methods
  private func startAudioClassification() {
    
    // Perform classification at regular intervals
    timer = Timer.scheduledTimer(withTimeInterval: 0.4, repeats: true, block: { timer in
        do {
          // Read the audio record's latest buffer into audioTensor's buffer.
          try self.audioTensor?.loadAudioRecord(audioRecord: self.audioRecord!)
          
          // Classify the resulting audio tensor buffer.
          if let audioTensor = self.audioTensor {
            let result =  try self.audioClassifier?.classify(audioTensor: audioTensor);
            if let categories = result?.classifications[0].categories {
              self.delegate?.soundClassifier(self, didClassifyWithCategories: Array(categories[0..<3]))
            }
          }
        }
        catch {
          print(error.localizedDescription)
        }
    })
  }

  deinit {
    timer?.invalidate()
  }
}
