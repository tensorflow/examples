// Copyright 2021 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

import AVFoundation
import UIKit
import os

final class ViewController: UIViewController {

  // MARK: Storyboards Connections
  @IBOutlet private weak var overlayView: OverlayView!
  @IBOutlet private weak var threadStepperLabel: UILabel!
  @IBOutlet private weak var threadStepper: UIStepper!
  @IBOutlet private weak var totalTimeLabel: UILabel!
  @IBOutlet private weak var scoreLabel: UILabel!
  @IBOutlet private weak var delegatesSegmentedControl: UISegmentedControl!
  @IBOutlet private weak var modelSegmentedControl: UISegmentedControl!

  // MARK: Pose estimation model configs
  private var modelType: ModelType = Constants.defaultModelType
  private var threadCount: Int = Constants.defaultThreadCount
  private var delegate: Delegates = Constants.defaultDelegate
  private let minimumScore = Constants.minimumScore

  // MARK: Visualization
  // Relative location of `overlayView` to `previewView`.
  private var imageViewFrame: CGRect?
  // Input image overlaid with the detected keypoints.
  var overlayImage: OverlayView?

  // MARK: Controllers that manage functionality
  // Handles all data preprocessing and makes calls to run inference.
  private var poseEstimator: PoseEstimator?
  private var cameraFeedManager: CameraFeedManager!

  // Serial queue to control all tasks related to the TFLite model.
  let queue = DispatchQueue(label: "serial_queue")

  // Flag to make sure there's only one frame processed at each moment.
  var isRunning = false

  // MARK: View Handling Methods
  override func viewDidLoad() {
    super.viewDidLoad()
    configSegmentedControl()
    configStepper()
    updateModel()
    configCameraCapture()
  }

  override func viewWillAppear(_ animated: Bool) {
    super.viewWillAppear(animated)
    cameraFeedManager?.startRunning()
  }

  override func viewWillDisappear(_ animated: Bool) {
    super.viewWillDisappear(animated)
    cameraFeedManager?.stopRunning()
  }

  override func viewDidLayoutSubviews() {
    super.viewDidLayoutSubviews()
    imageViewFrame = overlayView.frame
  }

  private func configCameraCapture() {
    cameraFeedManager = CameraFeedManager()
    cameraFeedManager.startRunning()
    cameraFeedManager.delegate = self
  }

  private func configStepper() {
    threadStepper.value = Double(threadCount)
    threadStepper.setDecrementImage(threadStepper.decrementImage(for: .normal), for: .normal)
    threadStepper.setIncrementImage(threadStepper.incrementImage(for: .normal), for: .normal)
  }

  private func configSegmentedControl() {
    // Set title for device control
    delegatesSegmentedControl.setTitleTextAttributes(
      [NSAttributedString.Key.foregroundColor: UIColor.lightGray],
      for: .normal)
    delegatesSegmentedControl.setTitleTextAttributes(
      [NSAttributedString.Key.foregroundColor: UIColor.black],
      for: .selected)
    // Remove existing segments to initialize it with `Delegates` entries.
    delegatesSegmentedControl.removeAllSegments()
    var defaultDelegateIndex = 0
    Delegates.allCases.enumerated().forEach { (index, eachDelegate) in
      if eachDelegate == delegate {
        defaultDelegateIndex = index
      }
      delegatesSegmentedControl.insertSegment(
        withTitle: eachDelegate.rawValue,
        at: index,
        animated: false)
    }
    delegatesSegmentedControl.selectedSegmentIndex = defaultDelegateIndex

    // Config model segment attributed
    modelSegmentedControl.setTitleTextAttributes(
      [NSAttributedString.Key.foregroundColor: UIColor.lightGray],
      for: .normal)
    modelSegmentedControl.setTitleTextAttributes(
      [NSAttributedString.Key.foregroundColor: UIColor.black],
      for: .selected)
    // Remove existing segments to initialize it with `Delegates` entries.
    modelSegmentedControl.removeAllSegments()
    var defaultModelTypeIndex = 0
    ModelType.allCases.enumerated().forEach { (index, eachModelType) in
      if eachModelType == modelType {
        defaultModelTypeIndex = index
      }
      modelSegmentedControl.insertSegment(
        withTitle: eachModelType.rawValue,
        at: index,
        animated: false)
    }
    modelSegmentedControl.selectedSegmentIndex = defaultModelTypeIndex
  }

  /// Call this method when there's change in pose estimation model config, including changing model
  /// or updating runtime config.
  private func updateModel() {
    // Update the model in the same serial queue with the inference logic to avoid race condition
    queue.async {
      do {
        switch self.modelType {
        case .posenet:
          self.poseEstimator = try PoseNet(
            threadCount: self.threadCount,
            delegate: self.delegate)
        case .movenetLighting, .movenetThunder:
          self.poseEstimator = try MoveNet(
            threadCount: self.threadCount,
            delegate: self.delegate,
            modelType: self.modelType)
        }
      } catch let error {
        os_log("Error: %@", log: .default, type: .error, String(describing: error))
      }
    }
  }

  @IBAction private func threadStepperValueChanged(_ sender: UIStepper) {
    threadCount = Int(sender.value)
    threadStepperLabel.text = "\(threadCount)"
    updateModel()
  }
  @IBAction private func delegatesValueChanged(_ sender: UISegmentedControl) {
    delegate = Delegates.allCases[sender.selectedSegmentIndex]
    updateModel()
  }

  @IBAction private func modelTypeValueChanged(_ sender: UISegmentedControl) {
    modelType = ModelType.allCases[sender.selectedSegmentIndex]
    updateModel()
  }
}

// MARK: - CameraFeedManagerDelegate Methods
extension ViewController: CameraFeedManagerDelegate {
  func cameraFeedManager(
    _ cameraFeedManager: CameraFeedManager, didOutput pixelBuffer: CVPixelBuffer
  ) {
    self.runModel(pixelBuffer)
  }

  /// Run pose estimation on the input frame from the camera.
  private func runModel(_ pixelBuffer: CVPixelBuffer) {
    // Guard to make sure that there's only 1 frame process at each moment.
    guard !isRunning else { return }

    // Guard to make sure that the pose estimator is already initialized.
    guard let estimator = poseEstimator else { return }

    // Run inference on a serial queue to avoid race condition.
    queue.async {
      self.isRunning = true
      defer { self.isRunning = false }

      // Run pose estimation
      do {
        let (result, times) = try estimator.estimateSinglePose(
            on: pixelBuffer)

        // Return to main thread to show detection results on the app UI.
        DispatchQueue.main.async {
          self.totalTimeLabel.text = String(format: "%.2fms",
                                            times.total * 1000)
          self.scoreLabel.text = String(format: "%.3f", result.score)

          // Allowed to set image and overlay
          let image = UIImage(ciImage: CIImage(cvPixelBuffer: pixelBuffer))

          // If score is too low, clear result remaining in the overlayView.
          if result.score < self.minimumScore {
            self.overlayView.image = image
            return
          }

          // Visualize the pose estimation result.
          self.overlayView.draw(at: image, person: result)
        }
      } catch {
        os_log("Error running pose estimation.", type: .error)
        return
      }
    }
  }
}

enum Constants {
  // Configs for the TFLite interpreter.
  static let defaultThreadCount = 4
  static let defaultDelegate: Delegates = .gpu
  static let defaultModelType: ModelType = .movenetThunder

  // Minimum score to render the result.
  static let minimumScore: Float32 = 0.2
}
