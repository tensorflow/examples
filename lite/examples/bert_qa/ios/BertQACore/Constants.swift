// Copyright 2020 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import UIKit

enum InterpreterOptions {
  // Default thread count is 2, unless maximum thread count is 1.
  static let threadCount = (
    defaultValue: 2,
    minimumValue: 1,
    maximumValue: Int(ProcessInfo.processInfo.activeProcessorCount),
    id: "threadCount"
  )
}

enum MobileBERT {
  static let maxAnsLen = 32
  static let maxQueryLen = 64
  static let maxSeqLen = 384

  static let predictAnsNum = 5
  static let outputOffset = 1  // Need to shift 1 for outputs ([CLS])

  static let doLowerCase = true

  static let inputDimension = [1, MobileBERT.maxSeqLen]
  static let outputDimension = [1, MobileBERT.maxSeqLen]

  static let dataset = File(name: "contents_from_squad_dict_format", ext: "json")
  static let vocabulary = File(name: "vocab", ext: "txt")
  static let model = File(name: "mobilebert_float_20191023", ext: "tflite")
}

struct File {
  let name: String
  let ext: String
  let description: String

  init(name: String, ext: String) {
    self.name = name
    self.ext = ext
    self.description = "\(name).\(ext)"
  }
}

enum CustomUI {
  static let textHighlightColor = UIColor(red: 1.0, green: 0.7, blue: 0.0, alpha: 0.3)

  static let runButtonOpacity = 0.8

  static let statusTextViewCornerRadius = CGFloat(7)
  static let suggestedQuestionCornerRadius = CGFloat(10)

  static let keyboardAnimationDuration = 0.23

  static let stackSpacing = CGFloat(5)
  static let padding = CGFloat(5)
  static let contentViewPadding = CGFloat(7)
  static let controlViewPadding = CGFloat(10)
  static let textSidePadding = CGFloat(4)
  static let textPadding = CGFloat(3)

  static let statusFontSize = CGFloat(14)
}

enum StatusMessage {
  static let askRun = "Tap ▶︎ button to get the answer."
  static let warnEmptyQuery = "⚠️Got empty question.\nPlease enter non-empty question."
  static let inferenceFailError = "❗️Failed to inference the answer."
}
