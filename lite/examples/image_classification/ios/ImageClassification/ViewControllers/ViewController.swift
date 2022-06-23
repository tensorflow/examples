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
  @IBOutlet weak var previewView: PreviewView!
  @IBOutlet weak var cameraUnavailableLabel: UILabel!
  @IBOutlet weak var resumeButton: UIButton!
  @IBOutlet weak var bottomSheetView: UIView!

  @IBOutlet weak var bottomSheetViewBottomSpace: NSLayoutConstraint!
  @IBOutlet weak var bottomSheetStateImageView: UIImageView!
  @IBOutlet weak var bottomViewHeightConstraint: NSLayoutConstraint!

  // MARK: Constants
  private let animationDuration = 0.5
  private let collapseTransitionThreshold: CGFloat = -40.0
  private let expandTransitionThreshold: CGFloat = 40.0
  private let delayBetweenInferencesMs = 1000.0

  // MARK: Instance Variables
  private let inferenceQueue = DispatchQueue(label: "org.tensorflow.lite.inferencequeue")
  private var previousInferenceTimeMs = Date.distantPast.timeIntervalSince1970 * 1000
  private var isInferenceQueueBusy = false
  private var initialBottomSpace: CGFloat = 0.0
  private var threadCount = DefaultConstants.threadCount
  private var maxResults = DefaultConstants.maxResults {
    didSet {
      guard let inferenceVC = inferenceViewController else { return }
      bottomViewHeightConstraint.constant = inferenceVC.collapsedHeight + 290
      view.layoutSubviews()
    }
  }
  private var scoreThreshold = DefaultConstants.scoreThreshold
  private var model: ModelType = .efficientnetLite0

  // MARK: Controllers that manage functionality
  // Handles all the camera related functionality
  private lazy var cameraCapture = CameraFeedManager(previewView: previewView)

  // Handles all data preprocessing and makes calls to run inference through the
  // `ImageClassificationHelper`.
  private var imageClassificationHelper: ImageClassificationHelper? =
    ImageClassificationHelper(
      modelFileInfo: DefaultConstants.model.modelFileInfo,
      threadCount: DefaultConstants.threadCount,
      resultCount: DefaultConstants.maxResults,
      scoreThreshold: DefaultConstants.scoreThreshold)

  // Handles the presenting of results on the screen
  private var inferenceViewController: InferenceViewController?

  // MARK: View Handling Methods
  override func viewDidLoad() {
    super.viewDidLoad()

    guard imageClassificationHelper != nil else {
      fatalError("Model initialization failed.")
    }

    cameraCapture.delegate = self
    addPanGesture()

    guard let inferenceVC = inferenceViewController else { return }
    bottomViewHeightConstraint.constant = inferenceVC.collapsedHeight + 290
    view.layoutSubviews()
  }

  override func viewWillAppear(_ animated: Bool) {
    super.viewWillAppear(animated)
    DispatchQueue.main.asyncAfter(deadline: .now() + 0.01) {
      self.changeBottomViewState()
    }
    #if !targetEnvironment(simulator)
      cameraCapture.checkCameraConfigurationAndStartSession()
    #endif
  }

  #if !targetEnvironment(simulator)
    override func viewWillDisappear(_ animated: Bool) {
      super.viewWillDisappear(animated)
      cameraCapture.stopSession()
    }
  #endif

  override var preferredStatusBarStyle: UIStatusBarStyle {
    return .lightContent
  }

  func presentUnableToResumeSessionAlert() {
    let alert = UIAlertController(
      title: "Unable to Resume Session",
      message: "There was an error while attempting to resume session.",
      preferredStyle: .alert
    )
    alert.addAction(UIAlertAction(title: "OK", style: .default, handler: nil))

    self.present(alert, animated: true)
  }

  // MARK: Storyboard Segue Handlers
  override func prepare(for segue: UIStoryboardSegue, sender: Any?) {
    super.prepare(for: segue, sender: sender)

    if segue.identifier == "EMBED" {
      inferenceViewController = segue.destination as? InferenceViewController
      inferenceViewController?.maxResults = maxResults
      inferenceViewController?.currentThreadCount = threadCount
      inferenceViewController?.delegate = self
    }
  }
}

// MARK: InferenceViewControllerDelegate Methods
extension ViewController: InferenceViewControllerDelegate {
  func viewController(
    _ viewController: InferenceViewController,
    needPerformActions action: InferenceViewController.Action
  ) {
    var isModelNeedsRefresh = false
    switch action {
    case .changeThreadCount(let threadCount):
      if self.threadCount != threadCount {
        isModelNeedsRefresh = true
      }
      self.threadCount = threadCount
    case .changeScoreThreshold(let scoreThreshold):
      if self.scoreThreshold != scoreThreshold {
        isModelNeedsRefresh = true
      }
      self.scoreThreshold = scoreThreshold
    case .changeMaxResults(let maxResults):
      if self.maxResults != maxResults {
        isModelNeedsRefresh = true
      }
      self.maxResults = maxResults
    case .changeModel(let model):
      if self.model != model {
        isModelNeedsRefresh = true
      }
      self.model = model
    }
    if isModelNeedsRefresh {
      imageClassificationHelper = ImageClassificationHelper(
        modelFileInfo: model.modelFileInfo,
        threadCount: threadCount,
        resultCount: maxResults,
        scoreThreshold: scoreThreshold
      )
    }
  }
}

// MARK: CameraFeedManagerDelegate Methods
extension ViewController: CameraFeedManagerDelegate {

  func didOutput(pixelBuffer: CVPixelBuffer) {
    // Make sure the model will not run too often, making the results changing quickly and hard to
    // read.
    let currentTimeMs = Date().timeIntervalSince1970 * 1000
    guard (currentTimeMs - previousInferenceTimeMs) >= delayBetweenInferencesMs else { return }
    previousInferenceTimeMs = currentTimeMs

    // Drop this frame if the model is still busy classifying a previous frame.
    guard !isInferenceQueueBusy else { return }

    inferenceQueue.async { [weak self] in
      guard let self = self else { return }

      self.isInferenceQueueBusy = true

      // Pass the pixel buffer to TensorFlow Lite to perform inference.
      let result = self.imageClassificationHelper?.classify(frame: pixelBuffer)

      self.isInferenceQueueBusy = false

      // Display results by handing off to the InferenceViewController.
      DispatchQueue.main.async {
        let resolution = CGSize(
          width: CVPixelBufferGetWidth(pixelBuffer), height: CVPixelBufferGetHeight(pixelBuffer))
        self.inferenceViewController?.inferenceResult = result
        self.inferenceViewController?.resolution = resolution
        self.inferenceViewController?.tableView.reloadData()
      }
    }
  }

  // MARK: Session Handling Alerts
  func sessionWasInterrupted(canResumeManually resumeManually: Bool) {

    // Updates the UI when session is interupted.
    if resumeManually {
      self.resumeButton.isHidden = false
    } else {
      self.cameraUnavailableLabel.isHidden = false
    }
  }

  func sessionInterruptionEnded() {
    // Updates UI once session interruption has ended.
    if !self.cameraUnavailableLabel.isHidden {
      self.cameraUnavailableLabel.isHidden = true
    }

    if !self.resumeButton.isHidden {
      self.resumeButton.isHidden = true
    }
  }

  func sessionRunTimeErrorOccured() {
    // Handles session run time error by updating the UI and providing a button if session can be
    // manually resumed.
    self.resumeButton.isHidden = false
    previewView.shouldUseClipboardImage = true
  }

  func presentCameraPermissionsDeniedAlert() {
    let alertController = UIAlertController(
      title: "Camera Permissions Denied",
      message:
        "Camera permissions have been denied for this app. You can change this by going to Settings",
      preferredStyle: .alert)

    let cancelAction = UIAlertAction(title: "Cancel", style: .cancel, handler: nil)
    let settingsAction = UIAlertAction(title: "Settings", style: .default) { (action) in
      UIApplication.shared.open(
        URL(string: UIApplication.openSettingsURLString)!, options: [:], completionHandler: nil)
    }
    alertController.addAction(cancelAction)
    alertController.addAction(settingsAction)

    present(alertController, animated: true, completion: nil)

    previewView.shouldUseClipboardImage = true
  }

  func presentVideoConfigurationErrorAlert() {
    let alert = UIAlertController(
      title: "Camera Configuration Failed", message: "There was an error while configuring camera.",
      preferredStyle: .alert)
    alert.addAction(UIAlertAction(title: "OK", style: .default, handler: nil))

    self.present(alert, animated: true)
    previewView.shouldUseClipboardImage = true
  }
}

// MARK: Bottom Sheet Interaction Methods
extension ViewController {

  // MARK: Bottom Sheet Interaction Methods
  /**
   This method adds a pan gesture to make the bottom sheet interactive.
   */
  private func addPanGesture() {
    let panGesture = UIPanGestureRecognizer(
      target: self, action: #selector(ViewController.didPan(panGesture:)))
    bottomSheetView.addGestureRecognizer(panGesture)
  }

  /** Change whether bottom sheet should be in expanded or collapsed state.
   */
  private func changeBottomViewState() {
    guard let inferenceVC = inferenceViewController else {
      return
    }

    if bottomSheetViewBottomSpace.constant == inferenceVC.collapsedHeight
      - bottomSheetView.bounds.size.height
    {
      bottomSheetViewBottomSpace.constant = 0.0
    } else {
      bottomSheetViewBottomSpace.constant =
        inferenceVC.collapsedHeight - bottomSheetView.bounds.size.height
    }
    setImageBasedOnBottomViewState()
  }

  /**
   Set image of the bottom sheet icon based on whether it is expanded or collapsed
   */
  private func setImageBasedOnBottomViewState() {
    if bottomSheetViewBottomSpace.constant == 0.0 {
      bottomSheetStateImageView.image = UIImage(named: "down_icon")
    } else {
      bottomSheetStateImageView.image = UIImage(named: "up_icon")
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
      initialBottomSpace = bottomSheetViewBottomSpace.constant
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
    guard
      bottomSpace <= 0.0
        && bottomSpace >= inferenceViewController!.collapsedHeight
          - bottomSheetView.bounds.size.height
    else {
      return
    }
    setBottomSheetLayout(withBottomSpace: bottomSpace)
  }

  /**
   This method changes bottom sheet state to either fully expanded or closed at the end of pan.
   */
  private func translateBottomSheetAtEndOfPan(withVerticalTranslation verticalTranslation: CGFloat)
  {
    // Changes bottom sheet state to either fully open or closed at the end of pan.
    let bottomSpace = bottomSpaceAtEndOfPan(withVerticalTranslation: verticalTranslation)
    setBottomSheetLayout(withBottomSpace: bottomSpace)
  }

  /**
   Return the final state of the bottom sheet view (whether fully collapsed or expanded) that is to
   be retained.
   */
  private func bottomSpaceAtEndOfPan(withVerticalTranslation verticalTranslation: CGFloat)
    -> CGFloat
  {
    // Calculates whether to fully expand or collapse bottom sheet when pan gesture ends.
    var bottomSpace = initialBottomSpace - verticalTranslation

    var height: CGFloat = 0.0
    if initialBottomSpace == 0.0 {
      height = bottomSheetView.bounds.size.height
    } else {
      height = inferenceViewController!.collapsedHeight
    }

    let currentHeight = bottomSheetView.bounds.size.height + bottomSpace

    if currentHeight - height <= collapseTransitionThreshold {
      bottomSpace = inferenceViewController!.collapsedHeight - bottomSheetView.bounds.size.height
    } else if currentHeight - height >= expandTransitionThreshold {
      bottomSpace = 0.0
    } else {
      bottomSpace = initialBottomSpace
    }

    return bottomSpace
  }

  /**
   This method layouts the change of the bottom space of bottom sheet with respect to the view
   managed by this controller.
   */
  func setBottomSheetLayout(withBottomSpace bottomSpace: CGFloat) {
    view.setNeedsLayout()
    bottomSheetViewBottomSpace.constant = bottomSpace
    view.setNeedsLayout()
  }

}

// Define default constants
enum DefaultConstants {
  static let threadCount = 4
  static let maxResults = 3
  static let scoreThreshold: Float = 0.2
  static let model: ModelType = .efficientnetLite0
}

/// TFLite model types
enum ModelType: CaseIterable {
  case efficientnetLite0
  case efficientnetLite1
  case efficientnetLite2
  case efficientnetLite3
  case efficientnetLite4

  var modelFileInfo: FileInfo {
    switch self {
    case .efficientnetLite0:
      return FileInfo("efficientnet_lite0", "tflite")
    case .efficientnetLite1:
      return FileInfo("efficientnet_lite1", "tflite")
    case .efficientnetLite2:
      return FileInfo("efficientnet_lite2", "tflite")
    case .efficientnetLite3:
      return FileInfo("efficientnet_lite3", "tflite")
    case .efficientnetLite4:
      return FileInfo("efficientnet_lite4", "tflite")
    }
  }

  var title: String {
    switch self {
    case .efficientnetLite0:
      return "EfficientNet-Lite0"
    case .efficientnetLite1:
      return "EfficientNet-Lite1"
    case .efficientnetLite2:
      return "EfficientNet-Lite2"
    case .efficientnetLite3:
      return "EfficientNet-Lite3"
    case .efficientnetLite4:
      return "EfficientNet-Lite4"
    }
  }
}
