//
//  BertQAClient.swift
//  BertQA
//
//  Created by sunit on 28/07/21.
//  Copyright © 2021 tensorflow. All rights reserved.
//

import Foundation
import TensorFlowLiteTaskText


class BertQAClient{
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
        
        let inferenceStartTime = Date()
        
        let answers:[TFLQAAnswer] = self.bertAnswerer.answer(context: content, question: query)
       
        let inferenceTime = Date().timeIntervalSince(inferenceStartTime) * 1000
        
        guard let answer = answers.first else {
            return nil
        }
        
        return Result(answer: answer, inferenceTime: inferenceTime)
      }
    
    
}
