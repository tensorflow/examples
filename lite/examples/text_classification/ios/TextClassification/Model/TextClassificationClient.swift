//
//  TextClassificationClient.swift
//  TextClassification
//
//  Created by Khurram Shehzad on 06/01/2020.
//  Copyright © 2020 Khurram Shehzad. All rights reserved.
//

import TensorFlowLite

import Foundation

typealias FileInfo = (name: String, extension: String)

let modelFileInfo = FileInfo(name: "text_classification", extension: "tflite")
let labelsFileInfo = FileInfo(name: "labels", extension: "txt")
let vocabFileInfo = FileInfo(name: "vocab", extension: "txt")

struct Result {
  let id: String
  let title: String
  let confidence: Float
}
final class TextClassificationnClient {
// The maximum length of an input sentence.
private let sentenceLength = 256
private let characterSet = CharacterSet(charactersIn: " ,.!?")
private let start = "<START>"
private let pad = "<PAD>"
private let unknown = "<UNKNOWN>"
/// TensorFlow Lite `Interpreter` object for performing inference on a given model.
private let interpreter: Interpreter
var dictionary = [String: Int]()
var labels = [String]()

/// A failable initializer for `TextClassificationnClient`. A new instance is created if the model, labels and vocab
/// files are successfully loaded from the app's main bundle.
init?(modelFileInfo: FileInfo, labelsFileInfo: FileInfo, vocabFileInfo: FileInfo) {
  guard let modelPath = Bundle.main.path(forResource: modelFileInfo.name, ofType: modelFileInfo.extension) else {
    print("Failed to load model file with name:\(modelFileInfo.name) and extension:\(modelFileInfo.extension)")
    return nil
  }
  do {
    interpreter = try Interpreter(modelPath: modelPath)
    // Allocate memory for the model's input `Tensor`s.
    try interpreter.allocateTensors()
  } catch {
    print("Failed to create interpreter with error:\(error)")
    return nil
  }
  loadLabels(fileInfo: labelsFileInfo)
  loadDictionary(fileInfo: vocabFileInfo)
}
  
func classify(text: String) -> [Result] {
  let input = tokenizeInputText(text: text)
  let data = Data(copyingBufferOf: input[0])
  do {
    try interpreter.copy(data, toInputAt: 0)
    try interpreter.invoke()
    let outputTensor = try interpreter.output(at: 0)
    if outputTensor.dataType == .float32 {
      let outputArray = [Float](unsafeData: outputTensor.data) ?? []
      var output = [Result]()
      for (index, label) in labels.enumerated() {
        output.append(Result(id: "", title: label, confidence: outputArray[index]))
      }
      output.sort(by: >)
      return output
    }
  } catch {
    print(error)
  }
  return []
}

func tokenizeInputText(text: String) -> [[Float]] {
  var temp = [Float](repeating: 0, count: sentenceLength)
  let array = text.split(characterSet: characterSet)
  var index = 0
  if let startValue = dictionary[start] {
    temp[index] = Float(startValue)
    index += 1
  }
  for word in array {
    if index >= sentenceLength {
      break
    }
    if let value = dictionary[word] {
      temp[index] = Float(value)
      index += 1
    } else if let value = dictionary[unknown] {
      temp[index] = Float(value)
      index += 1
    }
  }
  if let paddingValue = dictionary[pad] {
    let floatValue = Float(paddingValue)
    for i in index..<sentenceLength {
      temp[i] = floatValue
    }
  }
  return [temp]
}
  
/// Loads the labels from the labels file and stores them in the `labels` property.
private func loadLabels(fileInfo: FileInfo) {
  let filename = fileInfo.name
  let fileExtension = fileInfo.extension
  guard let fileURL = Bundle.main.url(forResource: filename, withExtension: fileExtension) else {
    fatalError("Labels file not found in bundle. Please add a labels file with name " +
                 "\(filename).\(fileExtension) and try again.")
  }
  do {
    let contents = try String(contentsOf: fileURL, encoding: .utf8)
    labels = contents.components(separatedBy: .newlines)
  } catch {
    fatalError("Labels file named \(filename).\(fileExtension) cannot be read. Please add a " +
                 "valid labels file and try again.")
  }
}

private func loadDictionary(fileInfo: FileInfo) {
    let filename = fileInfo.name
    let fileExtension = fileInfo.extension
    guard let fileURL = Bundle.main.url(forResource: filename, withExtension: fileExtension) else {
      fatalError("Vocab file not found in bundle. Please add a vocab file with name " +
                   "\(filename).\(fileExtension) and try again.")
    }
    do {
      let contents = try String(contentsOf: fileURL, encoding: .utf8)
      let lines = contents.components(separatedBy: .newlines)
      for line in lines {
        let components = line.components(separatedBy: .whitespaces)
        if components.count == 2,
          let i = Int(components[1]) {
          dictionary[components[0]] = i
        }
      }
    } catch {
      fatalError("Vocab file named \(filename).\(fileExtension) cannot be read. Please add a " +
                   "valid Vocab file and try again.")
    }
}
  
} // class TextClassificationnClient


extension Result: Comparable {
static func < (lhs: Result, rhs: Result) -> Bool {
  return lhs.confidence < rhs.confidence
}
} // extension Result

extension String {
func split(characterSet: CharacterSet) -> [String] {
  components(separatedBy: characterSet).filter { !$0.isEmpty }
}
} // extension String

extension Array {
/// Creates a new array from the bytes of the given unsafe data.
///
/// - Warning: The array's `Element` type must be trivial in that it can be copied bit for bit
///     with no indirection or reference-counting operations; otherwise, copying the raw bytes in
///     the `unsafeData`'s buffer to a new array returns an unsafe copy.
/// - Note: Returns `nil` if `unsafeData.count` is not a multiple of
///     `MemoryLayout<Element>.stride`.
/// - Parameter unsafeData: The data containing the bytes to turn into an array.
init?(unsafeData: Data) {
  guard unsafeData.count % MemoryLayout<Element>.stride == 0 else { return nil }
  #if swift(>=5.0)
  self = unsafeData.withUnsafeBytes { .init($0.bindMemory(to: Element.self)) }
  #else
  self = unsafeData.withUnsafeBytes {
    .init(UnsafeBufferPointer<Element>(
      start: $0,
      count: unsafeData.count / MemoryLayout<Element>.stride
    ))
  }
  #endif  // swift(>=5.0)
  }
} // extension Array

extension Data {
/// Creates a new buffer by copying the buffer pointer of the given array.
///
/// - Warning: The given array's element type `T` must be trivial in that it can be copied bit
///     for bit with no indirection or reference-counting operations; otherwise, reinterpreting
///     data from the resulting buffer has undefined behavior.
/// - Parameter array: An array with elements of type `T`.
init<T>(copyingBufferOf array: [T]) {
  self = array.withUnsafeBufferPointer(Data.init)
}
} // extension Data
