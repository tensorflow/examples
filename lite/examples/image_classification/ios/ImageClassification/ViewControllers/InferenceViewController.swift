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

  /**
   This method is called when the user changes the value to update model used for inference.
   **/
  func viewController(
    _ viewController: InferenceViewController,
    needPerformActions action: InferenceViewController.Action)
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
    case Results
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

  @IBOutlet weak var thresholdStepper: UIStepper!
  @IBOutlet weak var thresholdValueLabel: UILabel!

  @IBOutlet weak var maxResultStepper: UIStepper!
  @IBOutlet weak var maxResultLabel: UILabel!

  @IBOutlet weak var modelTextField: UITextField!

  // MARK: Constants
  private let normalCellHeight: CGFloat = 27.0
  private let separatorCellHeight: CGFloat = 42.0
  private let bottomSheetButtonDisplayHeight: CGFloat = 44.0
  private let minThreadCount = 1
  private let lightTextInfoColor = UIColor(
    displayP3Red: 117.0 / 255.0, green: 117.0 / 255.0, blue: 117.0 / 255.0, alpha: 1.0)
  private let infoFont = UIFont.systemFont(ofSize: 14.0, weight: .regular)
  private let highlightedFont = UIFont.systemFont(ofSize: 14.0, weight: .medium)

  // MARK: Instance Variables
  var inferenceResult: ImageClassificationResult? = nil
  var wantedInputWidth = 0
  var wantedInputHeight = 0
  var resolution = CGSize.zero
  var maxResults = DefaultConstants.maxResults
  var currentThreadCount = DefaultConstants.threadCount
  var scoreThreshold = DefaultConstants.scoreThreshold
  var modelSelectIndex = 0
  private var infoTextColor = UIColor.black

  private var modelSelect: ModelType {
    if modelSelectIndex < ModelType.allCases.count {
      return ModelType.allCases[modelSelectIndex]
    } else {
      return .efficientnetLite0
    }
  }

  // MARK: Delegate
  var delegate: InferenceViewControllerDelegate?

  // MARK: Computed properties
  var collapsedHeight: CGFloat {
    return normalCellHeight * CGFloat(maxResults - 1) + bottomSheetButtonDisplayHeight
  }

  override func viewDidLoad() {
    super.viewDidLoad()

    // Set up stepper
    threadStepper.isUserInteractionEnabled = true
    threadStepper.value = Double(currentThreadCount)

    // Set the info text color on iOS 11 and higher.
    if #available(iOS 11, *) {
      infoTextColor = UIColor(named: "darkOrLight")!
    }
    setupUI()
  }

  // MARK: private func
  private func setupUI() {
    threadStepper.value = Double(currentThreadCount)
    threadValueLabel.text = "\(currentThreadCount)"

    maxResultStepper.value = Double(maxResults)
    maxResultLabel.text = "\(maxResults)"

    thresholdStepper.value = Double(scoreThreshold)
    thresholdValueLabel.text = "\(scoreThreshold)"

    modelTextField.text = modelSelect.title

    let picker = UIPickerView()
    picker.delegate = self
    picker.dataSource = self
    modelTextField.inputView = picker

    let doneButton = UIButton(frame: CGRect(x: 0, y: 0, width: 60, height: 44))
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

  // MARK: Buttion Actions
  /**
   Delegate the change of number of threads to View Controller and change the stepper display.
   */
  @IBAction func onClickThreadStepper(_ sender: Any) {
    currentThreadCount = Int(threadStepper.value)
    threadValueLabel.text = "\(currentThreadCount)"
    delegate?.viewController(self, needPerformActions: .changeThreadCount(currentThreadCount))
  }

  @IBAction func thresholdStepperValueChanged(_ sender: UIStepper) {
    scoreThreshold = Float(sender.value)
    delegate?.viewController(self, needPerformActions: .changeScoreThreshold(scoreThreshold))
    thresholdValueLabel.text = "\(scoreThreshold)"
  }

  @IBAction func maxResultStepperValueChanged(_ sender: UIStepper) {
    maxResults = Int(sender.value)
    delegate?.viewController(self, needPerformActions: .changeMaxResults(maxResults))
    maxResultLabel.text = "\(maxResults)"
  }

  @objc
  func choseModelDoneButtonTouchUpInside(_ sender: UIButton) {
    delegate?.viewController(self, needPerformActions: .changeModel(modelSelect))
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
    case .Results:
      rowCount = maxResults
    case .InferenceInfo:
      rowCount = InferenceInfo.allCases.count
    }
    return rowCount
  }

  func tableView(_ tableView: UITableView, heightForRowAt indexPath: IndexPath) -> CGFloat {
    return normalCellHeight
  }

  func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
    let cell = tableView.dequeueReusableCell(withIdentifier: "INFO_CELL") as! InfoCell

    guard let inferenceSection = InferenceSections(rawValue: indexPath.section) else {
      return cell
    }

    var fieldName = ""
    var info = ""
    var font = infoFont
    var color = infoTextColor

    switch inferenceSection {
    case .Results:

      let tuple = displayStringsForResults(atRow: indexPath.row)
      fieldName = tuple.0
      info = tuple.1

      if indexPath.row == 0 {
        font = highlightedFont
        color = infoTextColor
      } else {
        font = infoFont
        color = lightTextInfoColor
      }

    case .InferenceInfo:
      let tuple = displayStringsForInferenceInfo(atRow: indexPath.row)
      fieldName = tuple.0
      info = tuple.1

    }
    cell.fieldNameLabel.font = font
    cell.fieldNameLabel.textColor = color
    cell.fieldNameLabel.text = fieldName
    cell.infoLabel.text = info
    return cell
  }

  // MARK: Format Display of information in the bottom sheet
  /**
   This method formats the display of the inferences for the current frame.
   */
  func displayStringsForResults(atRow row: Int) -> (String, String) {
    var fieldName: String = ""
    var info: String = ""

    guard let tempResult = inferenceResult, tempResult.classifications.categories.count > 0 else {

      if row == 1 {
        fieldName = "No Results"
        info = ""
      } else {
        fieldName = ""
        info = ""
      }
      return (fieldName, info)
    }

    if row < tempResult.classifications.categories.count {
      let category = tempResult.classifications.categories[row]
      fieldName = category.label ?? ""
      info = String(format: "%.2f", category.score * 100.0) + "%"
    } else {
      fieldName = ""
      info = ""
    }

    return (fieldName, info)
  }

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
      guard let finalResults = inferenceResult else {
        info = "0ms"
        break
      }
      info = String(format: "%.2fms", finalResults.inferenceTime)
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
