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

let modelFile = FileInfo(name: "", extension: "")

struct Result {
  let id: String
  let title: String
  let confidence: Float
}
final class TextClassificationnClient {
// The maximum length of an input sentence.
private let sentenceLength = 256
private let pattern = " |\\,|\\.|\\!|\\?|\n"
private let start = "<START>"
private let pad = "<PAD>"
private let unknown = "<UNKNOWN>"
/// TensorFlow Lite `Interpreter` object for performing inference on a given model.
private let interpreter: Interpreter
private var dictionary = [String: Int]()
private var labels = [String]()

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
  return []
}

func tokenizeInputText(text: String) -> [[Float]] {
  var temp = [Float]()
  let array = text.split(pattern: pattern)
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
    for _ in index..<sentenceLength {
      temp.append(floatValue)
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

extension String {
func split(pattern: String) -> [String] {
  do {
    let regularExpression = try NSRegularExpression(pattern: pattern, options: [])
    let results = regularExpression.matches(in: self, options: [], range: NSRange(location: 0, length: count))
    var components = [String]()
    for result in results {
      guard let range = Range(result.range, in: self) else { continue }
      components.append(String(self[range]))
    }
    return components
  } catch {
    print(error)
  }
  return []
}
} // extension String
