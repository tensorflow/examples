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

import UIKit

protocol InferenceViewDelegate: AnyObject {
  /// This method is called when the user changes the value to update model used for inference.
  func view(_ view: InferenceView, needPerformActions action: InferenceView.Action)
}

/// View to allows users changing the inference configs.
class InferenceView: UIView {

  enum Action {
    case changeModel(ModelType)
    case changeOverlap(Double)
    case changeMaxResults(Int)
    case changeScoreThreshold(Float)
    case changeThreadCount(Int)
  }

  weak var delegate: InferenceViewDelegate?

  @IBOutlet private weak var inferenceTimeLabel: UILabel!
  @IBOutlet private weak var overlabLabel: UILabel!
  @IBOutlet private weak var maxResulteLabel: UILabel!
  @IBOutlet private weak var thresholdLabel: UILabel!
  @IBOutlet private weak var threadsLabel: UILabel!
  @IBOutlet private weak var overLapStepper: UIStepper!
  @IBOutlet private weak var maxResultsStepper: UIStepper!
  @IBOutlet private weak var thresholdStepper: UIStepper!
  @IBOutlet private weak var threadsStepper: UIStepper!
  @IBOutlet private weak var modelSegmentedControl: UISegmentedControl!
  @IBOutlet private weak var showHidenButtonLayoutConstraint: NSLayoutConstraint!
  @IBOutlet private weak var showHidenButton: UIButton!

  /// Set the default settings.
  func setDefault(model: ModelType, overLab: Double, maxResult: Int, threshold: Float, threads: Int)
  {
    modelSegmentedControl.selectedSegmentIndex = model == .Yamnet ? 0 : 1
    overlabLabel.text = "\(Int(overLab * 100))%"
    overLapStepper.value = overLab
    maxResulteLabel.text = "\(maxResult)"
    maxResultsStepper.value = Double(maxResult)
    thresholdLabel.text = String(format: "%.1f", threshold)
    thresholdStepper.value = Double(threshold)
    threadsLabel.text = "\(threads)"
    threadsStepper.value = Double(threads)
  }

  func setInferenceTime(_ inferenceTime: TimeInterval) {
    inferenceTimeLabel.text = "\(Int(inferenceTime * 1000)) ms"
  }

  @IBAction func modelSegmentedValueChanged(_ sender: UISegmentedControl) {
    let modelSelect: ModelType = sender.selectedSegmentIndex == 0 ? .Yamnet : .speechCommandModel
    delegate?.view(self, needPerformActions: .changeModel(modelSelect))
  }

  @IBAction func overlapStepperValueChanged(_ sender: UIStepper) {
    overlabLabel.text = String(format: "%.0f", sender.value * 100) + "%"
    delegate?.view(self, needPerformActions: .changeOverlap(sender.value))
  }

  @IBAction func maxResultsStepperValueChanged(_ sender: UIStepper) {
    maxResulteLabel.text = "\(Int(sender.value))"
    delegate?.view(self, needPerformActions: .changeMaxResults(Int(sender.value)))
  }

  @IBAction func thresholdStepperValueChanged(_ sender: UIStepper) {
    thresholdLabel.text = String(format: "%.1f", sender.value)
    delegate?.view(self, needPerformActions: .changeScoreThreshold(Float(sender.value)))
  }

  @IBAction func threadsStepperValueChanged(_ sender: UIStepper) {
    threadsLabel.text = "\(Int(sender.value))"
    delegate?.view(self, needPerformActions: .changeThreadCount(Int(sender.value)))
  }

  @IBAction func showHidenButtonTouchUpInside(_ sender: UIButton) {
    sender.isSelected.toggle()
    showHidenButtonLayoutConstraint.constant = sender.isSelected ? 300 : 40
    UIView.animate(
      withDuration: 0.3,
      animations: {
        self.superview?.layoutIfNeeded()
      }, completion: nil)
  }
}
