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
import AVFoundation

public protocol AudioInputManagerDelegate: class {
  func audioInputManagerDidFailToAchievePermission(_ audioInputManager: AudioInputManager)
  func audioInputManager(_ audioInputManager: AudioInputManager, didCaptureChannelData: [Int16])
}

public class AudioInputManager {
  // MARK: - Constants

  private let sampleRate: Int
  private let recordingLength: Int
  private let audioUpdatePerSecond: Int
  private let conversionQueue = DispatchQueue(label: "conversionQueue")

  // MARK: - Variables
  public weak var delegate: AudioInputManagerDelegate?
  
  private var modelInputArray:Array<Int16>;
  private var modelInputArrayStartIndex = 0;

  private var audioEngine = AVAudioEngine()

  // MARK: - Methods

  public init(sampleRate: Int) {
    self.sampleRate = sampleRate
    self.recordingLength = sampleRate
    // Number of times the audio buffer should be updated per second
    // This value should be such that the audio engine bufferSize is in the range of [100, 400] ms as required by iOS audio engine
    // Updating audio 10 times per second will result in bufferSize of 100 ms
    self.audioUpdatePerSecond = 10
    self.modelInputArray = Array(repeating: 0, count: self.recordingLength)
  }

  public func checkPermissionsAndStartTappingMicrophone() {
    switch AVAudioSession.sharedInstance().recordPermission {
    case .granted:
      startTappingMicrophone()
    case .denied:
      delegate?.audioInputManagerDidFailToAchievePermission(self)
    case .undetermined:
      requestPermissions()
    @unknown default:
      fatalError()
    }
  }

  public func requestPermissions() {
    AVAudioSession.sharedInstance().requestRecordPermission { granted in
      if granted {
        self.startTappingMicrophone()
      } else {
        self.checkPermissionsAndStartTappingMicrophone()
      }
    }
  }

  /// Starts tapping the microphone input and converts it into the format for which the model is trained and
  /// periodically returns it in the block
  public func startTappingMicrophone() {
    let inputNode = audioEngine.inputNode
    let inputFormat = inputNode.outputFormat(forBus: 0)
    guard let recordingFormat = AVAudioFormat(
      commonFormat: .pcmFormatInt16,
      sampleRate: Double(sampleRate),
      channels: 1,
      interleaved: true
    ), let formatConverter = AVAudioConverter(from:inputFormat, to: recordingFormat) else { return }

    // installs a tap on the audio engine and specifying the buffer size and the input format.
    // The tap is triggered by the audio engine when the buffer is filled up.
    // It is required that the buffer size should be in the range of [100, 400] ms.
    inputNode.installTap(onBus: 0, bufferSize: AVAudioFrameCount(Int(inputFormat.sampleRate / self.audioUpdatePerSecond)), format: inputFormat) {
      buffer, _ in

      self.conversionQueue.async {
        // An AVAudioConverter is used to convert the microphone input to the format required
        // for the model.(pcm 16)
        guard let pcmBuffer = AVAudioPCMBuffer(
          pcmFormat: recordingFormat,
          frameCapacity: AVAudioFrameCount(recordingFormat.sampleRate * buffer.frameLength / AVAudioFrameCount(buffer.format.sampleRate)) // frameCapacity should be scaled appropriate to the recording format sample rate and audio engine's sample rate
        ) else { return }

        var error: NSError?
        let inputBlock: AVAudioConverterInputBlock = { _, outStatus in
          outStatus.pointee = AVAudioConverterInputStatus.haveData
          return buffer
        }

        formatConverter.convert(to: pcmBuffer, error: &error, withInputFrom: inputBlock)

        if let error = error {
          print(error.localizedDescription)
          return
        }
        if let channelData = pcmBuffer.int16ChannelData {
          let channelDataValue = channelData.pointee
          let channelDataValueArray = stride(
            from: 0,
            to: Int(pcmBuffer.frameLength),
            by: buffer.stride
          ).map { channelDataValue[$0] }

          // Find end index upto which audio data can be written in modelInputArray
          var modelInputArrayEndIndex = min(self.modelInputArrayStartIndex+Int(pcmBuffer!.frameLength), self.modelInputArray.count)
          // Update modelInputArray with the new channelDataValueArray in a round robin fashion
          self.modelInputArray[self.modelInputArrayStartIndex..<modelInputArrayEndIndex] = channelDataValueArray[0..<modelInputArrayEndIndex-self.modelInputArrayStartIndex];
          if self.modelInputArrayStartIndex+Int(pcmBuffer!.frameLength) > modelInputArrayEndIndex {
            self.modelInputArray[0..<self.modelInputArrayStartIndex+Int(pcmBuffer!.frameLength)-modelInputArrayEndIndex] = channelDataValueArray[modelInputArrayEndIndex-self.modelInputArrayStartIndex..<channelDataValueArray.count];
          }
          // Update the start index for the next iteration
          self.modelInputArrayStartIndex = (self.modelInputArrayStartIndex + Int(pcmBuffer!.frameLength)) % self.modelInputArray.count;
          
          // Copy audio data from modelInputArray such that the oldest audio sample is written at index 0 in timeCorrectedInputArray
          var timeCorrectedInputArray:[Int16] = Array(self.modelInputArray[self.modelInputArrayStartIndex..<self.modelInputArray.count])
          timeCorrectedInputArray.append(contentsOf: self.modelInputArray[0..<self.modelInputArrayStartIndex])

          // Converted pcm 16 values are delegated to the controller.
          self.delegate?.audioInputManager(self, didCaptureChannelData: timeCorrectedInputArray)
        }
      }
    }

    audioEngine.prepare()
    do {
      try audioEngine.start()
    } catch {
      print(error.localizedDescription)
    }
  }

}
