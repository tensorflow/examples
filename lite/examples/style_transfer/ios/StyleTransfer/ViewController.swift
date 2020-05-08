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

import UIKit
import os

class ViewController: UIViewController {

  /// Image picker for accessing the photo library or camera.
  private var imagePicker = UIImagePickerController()

  /// Style transferer instance reponsible for running the TF model. Uses a Int8-based model and
  /// runs inference on the CPU.
  private var cpuStyleTransferer: StyleTransferer?

  /// Style transferer instance reponsible for running the TF model. Uses a Float16-based model and
  /// runs inference on the GPU.
  private var gpuStyleTransferer: StyleTransferer?

  /// Target image to transfer a style onto.
  private var targetImage: UIImage?

  /// Style-representative image applied to the input image to create a pastiche.
  private var styleImage: UIImage?

  /// Style transfer result.
  private var styleTransferResult: StyleTransferResult?

  // UI elements
  @IBOutlet weak var imageView: UIImageView!
  @IBOutlet weak var photoCameraButton: UIButton!
  @IBOutlet weak var segmentedControl: UISegmentedControl!
  @IBOutlet weak var cropSwitch: UISwitch!
  @IBOutlet weak var useGPUSwitch: UISwitch!
  @IBOutlet weak var inferenceStatusLabel: UILabel!
  @IBOutlet weak var legendLabel: UILabel!
  @IBOutlet weak var styleImageView: UIImageView!
  @IBOutlet weak var runButton: UIButton!
  @IBOutlet weak var pasteImageButton: UIButton!

  override func viewDidLoad() {
    super.viewDidLoad()

    imageView.contentMode = .scaleAspectFill

    // Setup image picker.
    imagePicker.delegate = self
    imagePicker.sourceType = .photoLibrary

    // Set default style image.
    styleImage = StylePickerDataSource.defaultStyle()
    styleImageView.image = styleImage

    // Enable camera option only if current device has camera.
    let isCameraAvailable = UIImagePickerController.isCameraDeviceAvailable(.front)
      || UIImagePickerController.isCameraDeviceAvailable(.rear)
    if isCameraAvailable {
      photoCameraButton.isEnabled = true
    }

    // MetalDelegate is not available on iOS Simulator in Xcode versions below 11.
    // If you're not able to run GPU-based inference in iOS simulator, please check
    // your Xcode version.
    useGPUSwitch.isOn = true

    // Initialize new style transferer instances.
    StyleTransferer.newCPUStyleTransferer { result in
      switch result {
      case .success(let transferer):
        self.cpuStyleTransferer = transferer
      case .error(let wrappedError):
        print("Failed to initialize: \(wrappedError)")
      }
    }
    StyleTransferer.newGPUStyleTransferer { result in
      switch result {
      case .success(let transferer):
        self.gpuStyleTransferer = transferer
      case .error(let wrappedError):
        print("Failed to initialize: \(wrappedError)")
      }
    }
  }

  override func viewDidAppear(_ animated: Bool) {
    super.viewDidAppear(animated)
    // Observe foregrounding events for pasteboard access.
    addForegroundEventHandler()
    pasteImageButton.isEnabled = imageFromPasteboard() != nil
  }

  override func viewDidDisappear(_ animated: Bool) {
    super.viewDidDisappear(animated)
    NotificationCenter.default.removeObserver(self)
  }

  @IBAction func onTapPasteImage(_ sender: Any) {
    guard let image = imageFromPasteboard() else { return }
    let actionSheet = imageRoleSelectionAlert(image: image)
    present(actionSheet, animated: true, completion: nil)
  }

  @IBAction func onTapRunButton(_ sender: Any) {
    // Make sure that the cached target image is available.
    guard targetImage != nil else {
      self.inferenceStatusLabel.text = "Error: Input image is nil."
      return
    }

    runStyleTransfer(targetImage!)
  }

  @IBAction func onTapChangeStyleButton(_ sender: Any) {
    let pickerController = StylePickerViewController.fromStoryboard()
    pickerController.delegate = self
    present(pickerController, animated: true, completion: nil)
  }

  /// Open camera to allow user taking photo.
  @IBAction func onTapOpenCamera(_ sender: Any) {
    guard
      UIImagePickerController.isCameraDeviceAvailable(.front)
        || UIImagePickerController.isCameraDeviceAvailable(.rear)
    else {
      return
    }

    imagePicker.sourceType = .camera
    present(imagePicker, animated: true)
  }

  /// Open photo library for user to choose an image from.
  @IBAction func onTapPhotoLibrary(_ sender: Any) {
    imagePicker.sourceType = .photoLibrary
    present(imagePicker, animated: true)
  }

  /// Handle tapping on different display mode: Input, Style, Result
  @IBAction func onSegmentChanged(_ sender: Any) {
    switch segmentedControl.selectedSegmentIndex {
    case 0:
      // Mode 0: Show input image
      imageView.image = targetImage
    case 1:
      // Mode 1: Show style image
      imageView.image = styleImage
    case 2:
      // Mode 2: Show style transfer result.
      imageView.image = styleTransferResult?.resultImage
    default:
      break
    }
  }

  /// Handle changing center crop setting.
  @IBAction func onCropSwitchValueChanged(_ sender: Any) {
    // Make sure that the cached target image is available.
    guard targetImage != nil else {
      self.inferenceStatusLabel.text = "Error: Input image is nil."
      return
    }

    // Re-run style transfer upon center-crop setting changed.
    runStyleTransfer(targetImage!)
  }
}

// MARK: - Style Transfer

extension ViewController {
  /// Run style transfer on the given image, and show result on screen.
  ///  - Parameter image: The target image for style transfer.
  func runStyleTransfer(_ image: UIImage) {
    clearResults()

    let shouldUseQuantizedFloat16 = useGPUSwitch.isOn
    let transferer = shouldUseQuantizedFloat16 ? gpuStyleTransferer : cpuStyleTransferer

    // Make sure that the style transferer is initialized.
    guard let styleTransferer = transferer else {
      inferenceStatusLabel.text = "ERROR: Interpreter is not ready."
      return
    }

    guard let targetImage = self.targetImage else {
      inferenceStatusLabel.text = "ERROR: Select a target image."
      return
    }

    // Center-crop the target image if the user has enabled the option.
    let willCenterCrop = cropSwitch.isOn
    let image = willCenterCrop ? targetImage.cropCenter() : targetImage

    // Cache the potentially cropped image.
    self.targetImage = image

    // Show the potentially cropped image on screen.
    imageView.image = image

    // Make sure that the image is ready before running style transfer.
    guard image != nil else {
      inferenceStatusLabel.text = "ERROR: Image could not be cropped."
      return
    }

    guard let styleImage = styleImage else {
      inferenceStatusLabel.text = "ERROR: Select a style image."
      return
    }

    // Lock the crop switch and run buttons while style transfer is running.
    cropSwitch.isEnabled = false
    runButton.isEnabled = false

    // Run style transfer.
    styleTransferer.runStyleTransfer(
      style: styleImage,
      image: image!,
      completion: { result in
        // Show the result on screen
        switch result {
        case let .success(styleTransferResult):
          self.styleTransferResult = styleTransferResult

          // Change to show style transfer result
          self.segmentedControl.selectedSegmentIndex = 2
          self.onSegmentChanged(self)

          // Show result metadata
          self.showInferenceTime(styleTransferResult)
        case let .error(error):
          self.inferenceStatusLabel.text = error.localizedDescription
        }

        // Regardless of the result, re-enable switching between different display modes
        self.segmentedControl.isEnabled = true
        self.cropSwitch.isEnabled = true
        self.runButton.isEnabled = true
      })
  }

  /// Clear result from previous run to prepare for new style transfer.
  private func clearResults() {
    inferenceStatusLabel.text = "Running inference with TensorFlow Lite..."
    legendLabel.text = nil
    segmentedControl.isEnabled = false
    segmentedControl.selectedSegmentIndex = 0
  }

  /// Show processing time on screen.
  private func showInferenceTime(_ result: StyleTransferResult) {
    let timeString = "Preprocessing: \(Int(result.preprocessingTime * 1000))ms.\n"
      + "Style prediction: \(Int(result.stylePredictTime * 1000))ms.\n"
      + "Style transfer: \(Int(result.styleTransferTime * 1000))ms.\n"
      + "Post-processing: \(Int(result.postprocessingTime * 1000))ms.\n"

    inferenceStatusLabel.text = timeString
  }

}

// MARK: - UIImagePickerControllerDelegate

extension ViewController: UIImagePickerControllerDelegate, UINavigationControllerDelegate {

  func imagePickerController(
    _ picker: UIImagePickerController,
    didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey: Any]
  ) {

    if let pickedImage = info[.originalImage] as? UIImage {
      // Rotate target image to .up orientation to avoid potential orientation misalignment.
      guard let targetImage = pickedImage.transformOrientationToUp() else {
        inferenceStatusLabel.text = "ERROR: Image orientation couldn't be fixed."
        return
      }

      self.targetImage = targetImage

      if styleImage != nil {
        runStyleTransfer(targetImage)
      } else {
        imageView.image = targetImage
      }
    }

    dismiss(animated: true)
  }
}

// MARK: StylePickerViewControllerDelegate

extension ViewController: StylePickerViewControllerDelegate {

  func picker(_: StylePickerViewController, didSelectStyle image: UIImage) {
    styleImage = image
    styleImageView.image = image

    if let targetImage = targetImage {
      runStyleTransfer(targetImage)
    }
  }

}

// MARK: Pasteboard images

extension ViewController {

  fileprivate func imageFromPasteboard() -> UIImage? {
    return UIPasteboard.general.images?.first
  }

  fileprivate func imageRoleSelectionAlert(image: UIImage) -> UIAlertController {
    let controller = UIAlertController(title: "Paste Image",
                                       message: nil,
                                       preferredStyle: .actionSheet)
    controller.popoverPresentationController?.sourceView = view
    let setInputAction = UIAlertAction(title: "Set input image", style: .default) { _ in
      // Rotate target image to .up orientation to avoid potential orientation misalignment.
      guard let targetImage = image.transformOrientationToUp() else {
        self.inferenceStatusLabel.text = "ERROR: Image orientation couldn't be fixed."
        return
      }

      self.targetImage = targetImage
      self.imageView.image = targetImage
    }
    let setStyleAction = UIAlertAction(title: "Set style image", style: .default) { _ in
      guard let croppedImage = image.cropCenter() else {
        self.inferenceStatusLabel.text = "ERROR: Unable to crop style image."
        return
      }

      self.styleImage = croppedImage
      self.styleImageView.image = croppedImage
    }
    let cancelAction = UIAlertAction(title: "Cancel", style: .cancel) { _ in
      controller.dismiss(animated: true, completion: nil)
    }
    controller.addAction(setInputAction)
    controller.addAction(setStyleAction)
    controller.addAction(cancelAction)

    return controller
  }

  fileprivate func addForegroundEventHandler() {
    NotificationCenter.default.addObserver(self,
                                           selector: #selector(onForeground(_:)),
                                           name: UIApplication.willEnterForegroundNotification,
                                           object: nil)
  }

  @objc fileprivate func onForeground(_ sender: Any) {
    self.pasteImageButton.isEnabled = self.imageFromPasteboard() != nil
  }

}
