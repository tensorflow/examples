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

import AVFoundation
import UIKit

class ViewController: UIViewController {

  // MARK: Storyboards Connections
  @IBOutlet weak var bottomSheetView: CurvedView!
  @IBOutlet weak var previewView: PreviewView!
	  @IBOutlet weak var containerViewBottomSpace: NSLayoutConstraint!
  @IBOutlet weak var containerViewHeight: NSLayoutConstraint!
  @IBOutlet weak var cameraUnavailableLabel: UILabel!
  @IBOutlet weak var resumeButton: UIButton!
  @IBOutlet weak var upImageView: UIImageView!

  // MARK: Constants
  private let animationDuration = 0.5
  private let bottomSheetMinTransitionThreshold: CGFloat = 40.0

  // MARK: Instance Variables
  var initialBottomSpace: CGFloat = 0.0

  // MARK: Controllers that manage functionality
  // Handles all the camera related functionality
  private lazy var cameraCapture = CameraFeedManager(previewView: previewView)

  // Handles all data preprocessing and makes calls to run inference
  private let modelDataHandler: ModelDataHandler? = ModelDataHandler(modelFileName: "model", labelsFileName: "labels", labelsFileExtension: "txt")

  // Handles the presenting of results on the screen
  private var inferenceViewController: InferenceViewController?

  // MARK: View Handling Methods
  override func viewDidLoad() {
    super.viewDidLoad()

    // Checks if ModelDataHandler initialization was successful to determine if execution should be continued or not.
    guard modelDataHandler != nil else {
      fatalError("Model set up failed")
    }

    cameraCapture.delegate = self

    // Displays the gestures
    displayGestures()

    addPanGesture()

  }

  override func viewWillLayoutSubviews() {
    super.viewWillLayoutSubviews()
    guard let expandedHeight = inferenceViewController?.expandedHeight() else {
      return
    }
    containerViewHeight.constant = expandedHeight
  }

  override func viewWillAppear(_ animated: Bool) {
    super.viewWillAppear(animated)
    cameraCapture.checkCameraConfigurationAndStartSession()
    changeViewState()
  }

  override func viewWillDisappear(_ animated: Bool) {
    cameraCapture.stopSession()
  }

  override var preferredStatusBarStyle: UIStatusBarStyle {
    return .lightContent
  }

  // MARK: Storyboard Segue Handlers
  override func prepare(for segue: UIStoryboardSegue, sender: Any?) {
    super.prepare(for: segue, sender: sender)

    if segue.identifier == "EMBED" {

      guard let tempModelDataHandler = modelDataHandler else {
        return
      }
      inferenceViewController = segue.destination as? InferenceViewController
      inferenceViewController?.wantedInputHeight = tempModelDataHandler.wantedInputHeight
      inferenceViewController?.wantedInputWidth = tempModelDataHandler.wantedInputWidth
      inferenceViewController?.threadCountLimit = Int(tempModelDataHandler.threadCountLimit)
      inferenceViewController?.currentThreadCount = Int(tempModelDataHandler.threadCount)
      inferenceViewController?.delegate = self

    }
  }

  // MARK: Button Actions
  @IBAction func onClickResumeButton(_ sender: Any) {

    // Let's user resume session manually by button click
    cameraCapture.resumeInterruptedSession { (complete) in

      if complete {
        self.resumeButton.isHidden = true
        self.cameraUnavailableLabel.isHidden = true
      }
      else {
        self.presentUnableToResumeSessionAlert()
      }
    }
  }

  private func presentUnableToResumeSessionAlert() {
    let alert = UIAlertController(title: "Unable to Resume Session", message: "There was an error while attempting to resume session.", preferredStyle: .alert)
    alert.addAction(UIAlertAction(title: "OK", style: .default, handler: nil))

    present(alert, animated: true)
  }

  // MARK: Gesture Display Handling Methods

  /** This method displays the gestures on which the model could be trained. It enables only the gestures on which the model is trained. All the other gestures are shown disabled in the UI
   */
  private func displayGestures() {

    guard let tempModelDataHandler = modelDataHandler else {
      return
    }

    let labels = loadLabels(fromFileName: "display_labels", fileExtension: "txt")
    let gestures:[GestureDisplay]  = labels.map { ( label ) -> GestureDisplay in

      var isEnabled = true

      if !tempModelDataHandler.labels.contains(label.stringByTrimmingWhiteSpace()) {
        isEnabled = false
      }

      return GestureDisplay(name: label, isEnabled: isEnabled)
    }
    inferenceViewController?.displayGestures(gestures: gestures)
  }

  /**
   Loads the labels from the labels file and stores it in an instance variable
   */
  private func loadLabels(fromFileName fileName: String, fileExtension: String) -> [String] {

    var labels: [String] = []
    guard let fileURL = Bundle.main.url(forResource: fileName, withExtension: fileExtension) else {
      fatalError("Labels file not found in bundle. Please add a labels file with name \(fileName).\(fileExtension) and try again")
    }
    do {
      let contents = try String(contentsOf: fileURL, encoding: .utf8)
      labels = contents.components(separatedBy: .newlines)
      labels.removeAll { (label) -> Bool in
        return label == ""
      }
    }
    catch {
      fatalError("Labels file named \(fileName).\(fileExtension) cannot be read. Please add a valid labels file and try again.")
    }

    return labels
  }

}

// MARK: InferenceViewControllerDelegate Methods
extension ViewController: InferenceViewControllerDelegate {

  func didChangeThreadCount(to count: Int) {
    modelDataHandler?.set(numberOfThreads: Int32(count))
  }
}

// MARK: CameraFeedManagerDelegate Methods
extension ViewController: CameraFeedManagerDelegate {

  func didOutput(pixelBuffer: CVPixelBuffer) {

    let height = CVPixelBufferGetHeight(pixelBuffer)
    let width = CVPixelBufferGetWidth(pixelBuffer)

    // Runs the pixel buffer through the model.
    let result = modelDataHandler?.runModel(onFrame: pixelBuffer)

    DispatchQueue.main.async {
      // Hands off to the InferenceViewController to format and display results.
      self.inferenceViewController?.result = result
      self.inferenceViewController?.resolution = CGSize(width: width, height: height)
      self.inferenceViewController?.refreshResults()

    }
  }

  // MARK: Session Handling Alerts
  func sessionWasInterrupted(canResumeManually resumeManually: Bool) {

    // Updates the UI when session is interupted.
    if resumeManually == true {
      resumeButton.isHidden = false
    }
    else {
      cameraUnavailableLabel.isHidden = false
    }
  }

  func sessionInterruptionEnded() {

    // Updates UI once session interruption has ended.
    if !cameraUnavailableLabel.isHidden {
      cameraUnavailableLabel.isHidden = true
    }

    if !resumeButton.isHidden {
      resumeButton.isHidden = true
    }
  }

  func sessionRunTimeErrorOccured() {
    // Handles session run time error by updating the UI and providing a button if session can be manually resumed.
    resumeButton.isHidden = false
  }

  func presentCameraPermissionsDeniedAlert() {
    let alertController = UIAlertController(title: "Camera Permissions Denied", message: "Camera permissions have been denied for this app. You can change this by going to Settings", preferredStyle: .alert)

    let cancelAction = UIAlertAction(title: "Cancel", style: .cancel, handler: nil)
    let settingsAction = UIAlertAction(title: "Settings", style: .default) { (action) in
      UIApplication.shared.open(URL(string: UIApplication.openSettingsURLString)!, options: [:], completionHandler: nil)
    }

    alertController.addAction(cancelAction)
    alertController.addAction(settingsAction)

    present(alertController, animated: true, completion: nil)
  }

  func presentVideoConfigurationErrorAlert() {
    let alert = UIAlertController(title: "Camera Configuration Failed", message: "There was an error while configuring camera.", preferredStyle: .alert)
    alert.addAction(UIAlertAction(title: "OK", style: .default, handler: nil))

    present(alert, animated: true)
  }
}

// MARK: Bottom Sheet Interaction Methods
extension ViewController {

  // MARK: Bottom Sheet Interaction Methods
  /**
   This method adds a pan gesture to make the bottom sheet interactive.
   */
  private func addPanGesture() {
    let panGesture = UIPanGestureRecognizer(target: self, action: #selector(ViewController.didPan(panGesture:)))
    bottomSheetView.addGestureRecognizer(panGesture)
  }


  /** Change whether bottom sheet should be in expanded or collapsed state.
   */
  private func changeViewState() {

    guard let inferenceVC = inferenceViewController else {
      return
    }

    if containerViewBottomSpace.constant == inferenceVC.collapsedHeight - inferenceVC.expandedHeight() {

      containerViewBottomSpace.constant = 0.0
    }
    else {
      containerViewBottomSpace.constant =  inferenceVC.collapsedHeight - inferenceVC.expandedHeight()
    }
    setImageBasedOnBottomViewState()
  }

  /**
   Set image of the bottom sheet icon based on whether it is expanded or collapsed
   */
  private func setImageBasedOnBottomViewState() {

    if containerViewBottomSpace.constant == 0.0 {
      upImageView.image = UIImage(named: "down_icon")
    }
    else {
      upImageView.image = UIImage(named: "up_icon")
    }
  }

  /**
   This method responds to the user panning on the bottom sheet.
   */
  @objc func didPan(panGesture: UIPanGestureRecognizer) {

    // Opens or closes the bottom sheet based on the user's interaction with the bottom sheet.
    let translation = panGesture.translation(in: view)

    switch panGesture.state {
    case .began:
      initialBottomSpace = containerViewBottomSpace.constant
      translateBottomSheet(withVerticalTranslation: translation.y)
    case .changed:
      translateBottomSheet(withVerticalTranslation: translation.y)
    case .cancelled:
      setBottomSheetLayout(withBottomSpace: initialBottomSpace)
    case .ended:
      translateBottomSheetAtEndOfPan(withVerticalTranslation: translation.y)
      setImageBasedOnBottomViewState()
      initialBottomSpace = 0.0
    default:
      break
    }
  }

  /**
   This method sets bottom sheet translation while pan gesture state is continuously changing.
   */
  private func translateBottomSheet(withVerticalTranslation verticalTranslation: CGFloat) {

    let bottomSpace = initialBottomSpace - verticalTranslation
    guard bottomSpace <= 0.0 && bottomSpace >= inferenceViewController!.collapsedHeight - bottomSheetView.bounds.size.height else {
      return
    }
    setBottomSheetLayout(withBottomSpace: bottomSpace)
  }

  /**
   This method changes bottom sheet state to either fully expanded or closed at the end of pan.
   */
  private func translateBottomSheetAtEndOfPan(withVerticalTranslation verticalTranslation: CGFloat) {

    // Changes bottom sheet state to either fully open or closed at the end of pan.
    let bottomSpace = bottomSpaceAtEndOfPan(withVerticalTranslation: verticalTranslation)
    setBottomSheetLayout(withBottomSpace: bottomSpace)
  }

  /**
   Return the final state of the bottom sheet view (whether fully collapsed or expanded) that is to be retained.
   */
  private func bottomSpaceAtEndOfPan(withVerticalTranslation verticalTranslation: CGFloat) -> CGFloat {

    // Calculates whether to fully expand or collapse bottom sheet when pan gesture ends.
    var bottomSpace = initialBottomSpace - verticalTranslation

    var height: CGFloat = 0.0
    if initialBottomSpace == 0.0 {
      height = bottomSheetView.bounds.size.height
    }
    else {
      height = inferenceViewController!.collapsedHeight
    }

    let currentHeight = containerViewHeight.constant + bottomSpace

    if currentHeight - height <= -bottomSheetMinTransitionThreshold {
      bottomSpace = inferenceViewController!.collapsedHeight - containerViewHeight.constant
    }
    else if currentHeight - height >= bottomSheetMinTransitionThreshold {
      bottomSpace = 0.0
    }
    else {
      bottomSpace = initialBottomSpace
    }

    return bottomSpace
  }

  /**
   This method layouts the change of the bottom space of bottom sheet with respect to the view managed by this controller.
   */
  func setBottomSheetLayout(withBottomSpace bottomSpace: CGFloat) {

    view.layoutIfNeeded()
    containerViewBottomSpace.constant = bottomSpace
    view.layoutIfNeeded()
  }

}

