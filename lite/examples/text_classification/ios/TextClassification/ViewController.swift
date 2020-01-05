//
//  ViewController.swift
//  TextClassification
//
//  Created by Khurram Shehzad on 06/01/2020.
//  Copyright © 2020 Khurram Shehzad. All rights reserved.
//

import UIKit

private let modelFileInfo = FileInfo(name: "text_classification", extension: "tflite")
private let labelsFileInfo = FileInfo(name: "labels", extension: "txt")
private let vocabFileInfo = FileInfo(name: "vocab", extension: "txt")

class ViewController: UIViewController {
  
private var textClassificationClient: TextClassificationnClient?
  
override func viewDidLoad() {
  super.viewDidLoad()
  DispatchQueue.global().async {
    self.loadClient()
  }
}
  
private func loadClient() {
  textClassificationClient = TextClassificationnClient(modelFileInfo: modelFileInfo, labelsFileInfo: labelsFileInfo, vocabFileInfo: vocabFileInfo)
}
  
} // class ViewController
