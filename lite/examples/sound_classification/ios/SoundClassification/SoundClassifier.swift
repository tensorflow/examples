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
  private(set) var labelNames: [String] = []
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

//    setupInterpreter()
  }
  
  public func startAudioClassification() {
    
    // Perform classification at regular intervals
    timer = Timer.scheduledTimer(withTimeInterval: 0.4, repeats: true, block: { timer in
      DispatchQueue.global(qos: .background).async {
        do {
          // Read the audio record's latest buffer into audioTensor's buffer.
          try self.audioTensor?.loadAudioRecord(audioRecord: self.audioRecord!)
          
//          // Get the audio tensor buffer
//          let floatBuffer = self.audioTensor.ringBuffer.floatBuffer()
          
          // Classify the resulting audio tensor buffer.
          if let audioTensor = self.audioTensor {
            let result =  try self.audioClassifier?.classify(audioTensor: audioTensor);
            if let categories = result?.classifications[0].categories {
              self.delegate?.soundClassifier(self, didClassifyWithCategories: categories)
            }

          }

        }
        catch {
          print(error.localizedDescription)
        }
      }
    })
  }

//  /// Invokes the `Interpreter` and processes and returns the inference results.
//  public func start() {
//    let outputTensor: Tensor
//    do {
//
////      // This will be handled by Task Library.
////      let inputTensorData = Data(buffer: UnsafeBufferPointer(start: inputBuffer.data, count: Int(inputBuffer.size)))
////
////      try interpreter.copy(inputTensorData , toInputAt: audioBufferInputTensorIndex)
////      try interpreter.invoke()
////
////      outputTensor = try interpreter.output(at: 0)
//    } catch let error {
//      print(">>> Failed to invoke the interpreter with error: \(error.localizedDescription)")
//      return
//    }
//
//    // Gets the formatted and averaged results.
////    let probabilities = dataToFloatArray(outputTensor.data) ?? []
////    delegate?.soundClassifier(self, didInterpreteProbabilities: probabilities)
//  }

  // MARK: - Private Methods

//  private func setupAudioClassifier() {
//    guard let modelPath = Bundle.main.path(
//      forResource: modelFileName,
//      ofType: modelFileExtension
//    ) else { return }
//
//    do {
//      audioClassifier = AudioClassifier(modelPath:modelPath);
//
////      interpreter = try Interpreter(modelPath: modelPath)
////
////      try interpreter.allocateTensors()
////      let inputShape = try interpreter.input(at: 0).shape
////      sampleRate = inputShape.dimensions[1]
////
////      try interpreter.invoke()
////
////      labelNames = loadLabels()
//    } catch {
//      print("Failed to create the interpreter with error: \(error.localizedDescription)")
//      return
//    }
//  }

  private func loadLabels() -> [String] {
    guard let labelPath = Bundle.main.path(
      forResource: labelFilename,
      ofType: labelFileExtension
    ) else { return [] }

    var content = ""
    do {
      content = try String(contentsOfFile: labelPath, encoding: .utf8)
      let labels = content.components(separatedBy: "\n")
        .filter { !$0.isEmpty }
        .compactMap { line -> String in
          let splitPair = line.components(separatedBy: " ")
          let label = splitPair[1]
          let titleCasedLabel = label.components(separatedBy: "_")
            .compactMap { $0.capitalized }
            .joined(separator: " ")
          return titleCasedLabel
        }
      return labels
    } catch {
      print("Failed to load label content: '\(content)' with error: \(error.localizedDescription)")
      return []
    }
  }

  /// Creates a new buffer by copying the buffer pointer of the given `Int16` array.
  private func int16ArrayToData(_ buffer: [Int16]) -> Data {
    let floatData = buffer.map { Float($0) / Float(Int16.max) }
    
    return floatData.withUnsafeBufferPointer(Data.init)
  }
  

  /// Creates a new array from the bytes of the given unsafe data.
  /// - Returns: `nil` if `unsafeData.count` is not a multiple of `MemoryLayout<Float>.stride`.
  private func dataToFloatArray(_ data: Data) -> [Float]? {
    guard data.count % MemoryLayout<Float>.stride == 0 else { return nil }

    #if swift(>=5.0)
    return data.withUnsafeBytes { .init($0.bindMemory(to: Float.self)) }
    #else
    return data.withUnsafeBytes {
      .init(UnsafeBufferPointer<Float>(
        start: $0,
        count: unsafeData.count / MemoryLayout<Element>.stride
      ))
    }
    #endif // swift(>=5.0)
  }
  
  deinit {
    timer?.invalidate()
  }
}
