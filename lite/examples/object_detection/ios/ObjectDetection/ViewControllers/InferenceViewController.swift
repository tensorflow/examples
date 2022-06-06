// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

// MARK: InferenceViewControllerDelegate Method Declarations
protocol InferenceViewControllerDelegate {

  /// Called when the user has changed the model settings. The delegate needs to subsequently
  /// perform the action on the TFLite model.
  func viewController(
    _ viewController: InferenceViewController,
    didReceiveAction action: InferenceViewController.Action)
}

class InferenceViewController: UIViewController {

  enum Action {
    case changeThreadCount(Int)
    case changeScoreThreshold(Float)
    case changeMaxResults(Int)
    case changeModel(ModelType)
  }

  // MARK: Sections and Information to display
  private enum InferenceSections: Int, CaseIterable {
    case InferenceInfo
  }

  private enum InferenceInfo: Int, CaseIterable {
    case Resolution
    case InferenceTime

    func displayString() -> String {

      var toReturn = ""

      switch self {
      case .Resolution:
        toReturn = "Resolution"
      case .InferenceTime:
        toReturn = "Inference Time"

      }
      return toReturn
    }
  }

  // MARK: Storyboard Outlets
  @IBOutlet weak var tableView: UITableView!
  @IBOutlet weak var threadStepper: UIStepper!
  @IBOutlet weak var threadValueLabel: UILabel!
  @IBOutlet weak var maxResultStepper: UIStepper!
  @IBOutlet weak var maxResultLabel: UILabel!
  @IBOutlet weak var thresholdStepper: UIStepper!
  @IBOutlet weak var thresholdLabel: UILabel!
  @IBOutlet weak var modelTextField: UITextField!

  // MARK: Constants
  private let normalCellHeight: CGFloat = 27.0
  private let separatorCellHeight: CGFloat = 42.0
  private let bottomSpacing: CGFloat = 21.0
  private let bottomSheetButtonDisplayHeight: CGFloat = 60.0
  private let infoTextColor = UIColor.black
  private let lightTextInfoColor = UIColor(
    displayP3Red: 117.0 / 255.0, green: 117.0 / 255.0, blue: 117.0 / 255.0, alpha: 1.0)
  private let infoFont = UIFont.systemFont(ofSize: 14.0, weight: .regular)
  private let highlightedFont = UIFont.systemFont(ofSize: 14.0, weight: .medium)

  // MARK: Instance Variables
  var inferenceTime: Double = 0
  var resolution: CGSize = CGSize.zero
  var currentThreadCount: Int = 0
  var scoreThreshold: Float = 0.5
  var maxResults: Int = 3
  var modelSelectIndex: Int = 0

  private var modelSelect: ModelType {
    if modelSelectIndex < ModelType.allCases.count {
      return ModelType.allCases[modelSelectIndex]
    } else {
      return .ssdMobileNetV1
    }
  }

  // MARK: Delegate
  var delegate: InferenceViewControllerDelegate?

  // MARK: Computed properties
  var collapsedHeight: CGFloat {
    return bottomSheetButtonDisplayHeight

  }

  override func viewDidLoad() {
    super.viewDidLoad()
    setupUI()
  }
  // MARK: private func
  private func setupUI() {

    threadStepper.value = Double(currentThreadCount)
    threadValueLabel.text = "\(currentThreadCount)"

    maxResultStepper.value = Double(maxResults)
    maxResultLabel.text = "\(maxResults)"

    thresholdStepper.value = Double(scoreThreshold)
    thresholdLabel.text = "\(scoreThreshold)"

    modelTextField.text = modelSelect.title

    let picker = UIPickerView()
    picker.delegate = self
    picker.dataSource = self
    modelTextField.inputView = picker

    let doneButton = UIButton(
      frame: CGRect(
        x: 0,
        y: 0,
        width: 60,
        height: 44))
    doneButton.setTitle("Done", for: .normal)
    doneButton.setTitleColor(.blue, for: .normal)
    doneButton.addTarget(
      self, action: #selector(choseModelDoneButtonTouchUpInside(_:)), for: .touchUpInside)
    let inputAccessoryView = UIView(
      frame: CGRect(
        x: 0,
        y: 0,
        width: UIScreen.main.bounds.size.width,
        height: 44))
    inputAccessoryView.backgroundColor = .gray
    inputAccessoryView.addSubview(doneButton)
    modelTextField.inputAccessoryView = inputAccessoryView
  }

  // MARK: Button Actions
  /// Delegate the change of number of threads to ViewController and change the stepper display.

  @IBAction func threadStepperValueChanged(_ sender: UIStepper) {
    currentThreadCount = Int(sender.value)
    delegate?.viewController(self, didReceiveAction: .changeThreadCount(currentThreadCount))
    threadValueLabel.text = "\(currentThreadCount)"
  }

  @IBAction func thresholdStepperValueChanged(_ sender: UIStepper) {
    scoreThreshold = Float(sender.value)
    delegate?.viewController(self, didReceiveAction: .changeScoreThreshold(scoreThreshold))
    thresholdLabel.text = "\(scoreThreshold)"
  }

  @IBAction func maxResultStepperValueChanged(_ sender: UIStepper) {
    maxResults = Int(sender.value)
    delegate?.viewController(self, didReceiveAction: .changeMaxResults(maxResults))
    maxResultLabel.text = "\(maxResults)"
  }

  @objc
  func choseModelDoneButtonTouchUpInside(_ sender: UIButton) {
    delegate?.viewController(self, didReceiveAction: .changeModel(modelSelect))
    modelTextField.text = modelSelect.title
    modelTextField.resignFirstResponder()
  }

}

// MARK: UITableView Data Source
extension InferenceViewController: UITableViewDelegate, UITableViewDataSource {

  func numberOfSections(in tableView: UITableView) -> Int {

    return InferenceSections.allCases.count
  }

  func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {

    guard let inferenceSection = InferenceSections(rawValue: section) else {
      return 0
    }

    var rowCount = 0
    switch inferenceSection {
    case .InferenceInfo:
      rowCount = InferenceInfo.allCases.count
    }
    return rowCount
  }

  func tableView(_ tableView: UITableView, heightForRowAt indexPath: IndexPath) -> CGFloat {

    var height: CGFloat = 0.0

    guard let inferenceSection = InferenceSections(rawValue: indexPath.section) else {
      return height
    }

    switch inferenceSection {
    case .InferenceInfo:
      if indexPath.row == InferenceInfo.allCases.count - 1 {
        height = separatorCellHeight + bottomSpacing
      } else {
        height = normalCellHeight
      }
    }
    return height
  }

  func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {

    let cell = tableView.dequeueReusableCell(withIdentifier: "INFO_CELL") as! InfoCell

    guard let inferenceSection = InferenceSections(rawValue: indexPath.section) else {
      return cell
    }

    var fieldName = ""
    var info = ""

    switch inferenceSection {
    case .InferenceInfo:
      let tuple = displayStringsForInferenceInfo(atRow: indexPath.row)
      fieldName = tuple.0
      info = tuple.1

    }
    cell.fieldNameLabel.font = infoFont
    cell.fieldNameLabel.textColor = infoTextColor
    cell.fieldNameLabel.text = fieldName
    cell.infoLabel.text = info
    return cell
  }

  // MARK: Format Display of information in the bottom sheet
  /**
   This method formats the display of additional information relating to the inferences.
   */
  func displayStringsForInferenceInfo(atRow row: Int) -> (String, String) {

    var fieldName: String = ""
    var info: String = ""

    guard let inferenceInfo = InferenceInfo(rawValue: row) else {
      return (fieldName, info)
    }

    fieldName = inferenceInfo.displayString()

    switch inferenceInfo {
    case .Resolution:
      info = "\(Int(resolution.width))x\(Int(resolution.height))"
    case .InferenceTime:

      info = String(format: "%.2fms", inferenceTime)
    }

    return (fieldName, info)
  }
}

extension InferenceViewController: UIPickerViewDelegate, UIPickerViewDataSource {
  func numberOfComponents(in pickerView: UIPickerView) -> Int {
    return 1
  }

  func pickerView(_ pickerView: UIPickerView, numberOfRowsInComponent component: Int) -> Int {
    return ModelType.allCases.count
  }

  func pickerView(_ pickerView: UIPickerView, titleForRow row: Int, forComponent component: Int)
    -> String?
  {
    if row < ModelType.allCases.count {
      return ModelType.allCases[row].title
    } else {
      return nil
    }
  }

  func pickerView(_ pickerView: UIPickerView, didSelectRow row: Int, inComponent component: Int) {
    modelSelectIndex = row
  }
}

class InfoCell: UITableViewCell {
  @IBOutlet weak var fieldNameLabel: UILabel!
  @IBOutlet weak var infoLabel: UILabel!
}
