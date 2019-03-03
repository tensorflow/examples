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
   This method is called when the user changes the stepper value to update number of threads used for inference.
   */
  func didChangeThreadCount(to count: Int)
}

/**This is used to represent the display name of a gesture and whether or not it is part of the classes on which the model is trained.
 */
struct GestureDisplay {
  let name: String
  let isEnabled: Bool
}

class InferenceViewController: UIViewController {

  // MARK: Information to display

  private enum InferenceInfo: Int, CaseIterable {
    case Resolution
    case Crop
    case InferenceTime

    func displayString() -> String {

      var toReturn = ""

      switch self {
      case .Resolution:
        toReturn = "Resolution"
      case .Crop:
        toReturn = "Crop"
      case .InferenceTime:
        toReturn = "Inference Time"

      }
      return toReturn
    }
  }

  // MARK: Storyboard Outlets
  @IBOutlet weak var gestureCollectionViewHeight: NSLayoutConstraint!
  @IBOutlet weak var stepperView: UIView!
  @IBOutlet weak var tableView: UITableView!
  @IBOutlet weak var threadStepper: UIStepper!
  @IBOutlet weak var stepperValueLabel: UILabel!
  @IBOutlet weak var gestureCollectionView: UICollectionView!
  @IBOutlet weak var tableViewHeightConstraint: NSLayoutConstraint!

  // MARK: Constants
  private var normalCellHeight: CGFloat {
   return lableHeight + labeltopSpacing * 2
  }

  private let lableHeight: CGFloat = 17.0
  private let labeltopSpacing: CGFloat = 5.0
  private let tableViewCollectionViewVerticalSpacing: CGFloat = 27.0
  private let separatorCellHeight: CGFloat = 42.0
  private let bottomSpacing: CGFloat = 21.0
  private let minThreadCount = 1
  private let bottomSheetButtonDisplayHeight: CGFloat = 44.0
  private let infoTextColor = UIColor.black
  private let lightTextInfoColor = UIColor(displayP3Red: 117.0/255.0, green: 117.0/255.0, blue: 117.0/255.0, alpha: 1.0)
  private let infoFont = UIFont.systemFont(ofSize: 14.0, weight: .regular)
  private let highlightedFont = UIFont.systemFont(ofSize: 14.0, weight: .medium)
  private let collectionViewPadding: CGFloat = 15.0
  private let heightPadding: CGFloat = 1.0
  private let itemsInARow: CGFloat = 4.0
  private let imageInset: CGFloat = 8.0
  private let stepperSuperViewHeight: CGFloat = 53.0

  var gestures: [GestureDisplay] = []
  var result: Result?

  var collectionViewHeight: CGFloat {
    get {
      let collectionViewWidth = self.view.bounds.size.width - (2 * collectionViewPadding)
      let itemWidth = (collectionViewWidth - ((itemsInARow - 1) * collectionViewPadding)) / itemsInARow

      return (itemWidth * 2.0) + collectionViewPadding + heightPadding
    }
  }

  // MARK: Instance Variables
  var inferenceResult: Result? = nil
  var wantedInputWidth: Int = 0
  var wantedInputHeight: Int = 0
  var resolution: CGSize = CGSize.zero
  var maxResults: Int = 0
  var threadCountLimit: Int = 0
  var currentThreadCount: Int = 0
  private var highlightedRow = -1

  // MARK: Delegate
  var delegate: InferenceViewControllerDelegate?

  // MARK: Computed properties
  var collapsedHeight: CGFloat {
    return collectionViewHeight + bottomSheetButtonDisplayHeight + tableViewCollectionViewVerticalSpacing

  }

  private var tableViewHeight: CGFloat {

    return CGFloat(InferenceInfo.allCases.count - 1) * normalCellHeight + lableHeight + separatorCellHeight + labeltopSpacing
  }

  override func viewDidLoad() {
    super.viewDidLoad()

    // Set up stepper
    threadStepper.isUserInteractionEnabled = true
    threadStepper.maximumValue = Double(threadCountLimit)
    threadStepper.minimumValue = Double(minThreadCount)
    threadStepper.value = Double(currentThreadCount)
    gestureCollectionView.reloadData()

  }

  override func viewWillLayoutSubviews() {
    super.viewWillLayoutSubviews()

    gestureCollectionViewHeight.constant = collectionViewHeight
    tableViewHeightConstraint.constant = tableViewHeight
  }

  func expandedHeight() -> CGFloat {

    return collapsedHeight + tableViewHeight + stepperSuperViewHeight
  }

  // MARK: Gesture Highlight Methods
  /**
   Displays the gestures specified in the parameter.
   */
  func displayGestures(gestures: [GestureDisplay]) {
    self.gestures = gestures
    highlightGesture(with: -1)
  }

  /**
   Refreshes the results in collectionview by highlighting currently identified gesture.
   */
  func refreshResults() {

    highlightIdentifiedGesture()
    self.tableView.reloadData()

  }

  /**Highlights the currently identified gesture by mapping the name of top inference with the gestures that are displayed.
   */
  private func highlightIdentifiedGesture() {

    guard let inferences = result?.inferences, inferences.count > 0 else {
      return
    }

    guard  let index = self.gestures.index(where: {$0.name.stringByTrimmingWhiteSpace() == inferences[0].className}) else {
      return
    }

    highlightGesture(with: index)

  }

  /**Highlights the currently identified gesture.
   */
  func highlightGesture(with index: Int) {

    self.highlightedRow = index
    self.gestureCollectionView.reloadData()
  }

  // MARK: Buttion Actions
  /**
   Delegate the change of number of threads to View Controller and change the stepper display.
   */
  @IBAction func onClickThreadStepper(_ sender: Any) {

    delegate?.didChangeThreadCount(to: Int(threadStepper.value))
    currentThreadCount = Int(threadStepper.value)
    stepperValueLabel.text = "\(currentThreadCount)"
  }
}

// MARK: UITableView Data Source
extension InferenceViewController: UITableViewDelegate, UITableViewDataSource {

  func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {

    return InferenceInfo.allCases.count
  }

  func tableView(_ tableView: UITableView, heightForRowAt indexPath: IndexPath) -> CGFloat {

    let height = indexPath.row == InferenceInfo.allCases.count - 1 ? (separatorCellHeight + labeltopSpacing + lableHeight) : normalCellHeight

    return height
  }

  func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {

    let cell = tableView.dequeueReusableCell(withIdentifier: "INFO_CELL") as! InfoCell

    var fieldName = ""
    var info = ""

    let tuple = displayStringsForInferenceInfo(atRow: indexPath.row)
    fieldName = tuple.0
    info = tuple.1

    cell.fieldNameLabel.font = infoFont
    cell.fieldNameLabel.textColor = infoTextColor
    cell.fieldNameLabel.text = fieldName
    cell.infoLabel.text = info
    return cell
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
    case .Crop:
      info = "\(wantedInputWidth)x\(wantedInputHeight)"
    case .InferenceTime:
      guard let finalResults = result else {
        info = "0ms"
        break
      }
      info = String(format: "%.2fms", finalResults.inferenceTime)
    }

    return(fieldName, info)
  }
}

// MARK: UICollectionView Data Source and Delegate
extension InferenceViewController: UICollectionViewDataSource, UICollectionViewDelegate, UICollectionViewDelegateFlowLayout {

  func numberOfSections(in collectionView: UICollectionView) -> Int {
    return 1
  }

  func collectionView(_ collectionView: UICollectionView, numberOfItemsInSection section: Int) -> Int {

    return gestures.count
  }

  func collectionView(_ collectionView: UICollectionView, cellForItemAt indexPath: IndexPath) -> UICollectionViewCell {

    let cell = collectionView.dequeueReusableCell(withReuseIdentifier: "GESTURE_CELL", for: indexPath) as! GestureCollectionViewCell

    let gesture = gestures[indexPath.item]
    var selectionImage: UIImage?

    let imageName = gesture.name.lowercased().replacingOccurrences(of: " ", with: "_")
    cell.iconImageView.image = UIImage(named: imageName)

    cell.nameLabel.text = gesture.name.uppercased().replacingOccurrences(of: " ", with: "\n")

    // Shows gestures which are not part of the trained set as disabled.
    cell.alpha = gesture.isEnabled ? 1.0 : 0.3

    // Highlights gesture currently identified.
    if indexPath.item == highlightedRow {
      selectionImage = UIImage(named: "selection_base")?.resizableImage(withCapInsets: UIEdgeInsets(top: imageInset, left: imageInset, bottom: imageInset, right: imageInset), resizingMode: .stretch)
    }
    else {
      selectionImage = UIImage(named: "selection_base_default")?.resizableImage(withCapInsets: UIEdgeInsets(top: imageInset, left: imageInset, bottom: imageInset, right: imageInset), resizingMode: .stretch)
    }
    cell.selectionImageView.image = selectionImage

    return  cell
  }

  func collectionView(_ collectionView: UICollectionView, layout collectionViewLayout: UICollectionViewLayout, sizeForItemAt indexPath: IndexPath) -> CGSize {

    let collectionViewWidth = self.view.bounds.size.width - 2 * collectionViewPadding
    let itemWidth = (collectionViewWidth - ((itemsInARow - 1) * collectionViewPadding)) / itemsInARow
    let size = CGSize(width: itemWidth, height: itemWidth)

    return size
  }
}


