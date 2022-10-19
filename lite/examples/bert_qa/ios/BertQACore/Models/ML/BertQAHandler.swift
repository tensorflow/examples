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

import Foundation
import TensorFlowLiteTaskText
import os

/// Handles Bert model with TensorFlow Lite.
final class BertQAHandler {
    private var bertAnswerer: TFLBertQuestionAnswerer
    init(){
        let modelFile: File = MobileBERT.model
        
        guard let modelPath = Bundle.main.path(
                forResource: modelFile.name, ofType: modelFile.ext)
        else {
            fatalError("Failed to load the model file: \(modelFile.description)")
        }
        
        self.bertAnswerer = TFLBertQuestionAnswerer.questionAnswerer(modelPath: modelPath)
    }
    
    func run(query: String, content: String) -> Result? {
        // MARK: - Inferencing
        let inferenceStartTime = Date()
        
        let answers:[TFLQAAnswer] = self.bertAnswerer.answer(context: content, question: query)
        
        let inferenceTime = Date().timeIntervalSince(inferenceStartTime) * 1000
        
        guard let answer = answers.first else {
            return nil
        }
        
        return Result(answer: answer, inferenceTime: inferenceTime)
    }
}
