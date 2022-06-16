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

class ViewController: UIViewController {

  /// Image picker for accessing the photo library or camera.
  private var imagePicker = UIImagePickerController()

  /// Image segmentator instance that runs image segmentation.
  private var imageSegmentationHelper: ImageSegmentationHelper?

  /// Target image to run image segmentation on.
  private var targetImage: UIImage?

  /// Processed (e.g center cropped)) image from targetImage that is fed to imageSegmentator.
  private var segmentationInput: UIImage?

  /// Image segmentation result.
  private var segmentationResult: ImageSegmentationResult?

  /// UI elements
  @IBOutlet weak var inputImageView: UIImageView!
  @IBOutlet weak var resultImageView: UIImageView!

  @IBOutlet weak var photoCameraButton: UIButton!
  @IBOutlet weak var segmentedControl: UISegmentedControl!
  @IBOutlet weak var cropSwitch: UISwitch!
  @IBOutlet weak var inferenceStatusLabel: UILabel!
  @IBOutlet weak var legendLabel: UILabel!

  override func viewDidLoad() {
    super.viewDidLoad()

    // Setup image picker.
    imagePicker.delegate = self
    imagePicker.sourceType = .photoLibrary

    // Enable camera option only if current device has camera.
    let isCameraAvailable =
      UIImagePickerController.isCameraDeviceAvailable(.front)
      || UIImagePickerController.isCameraDeviceAvailable(.rear)
    if isCameraAvailable {
      photoCameraButton.isEnabled = true
    }

    // Initialize an image segmentator instance.
    ImageSegmentationHelper.newInstance { result in
      switch result {
      case let .success(segmentationHelper):
        // Store the initialized instance for use.
        self.imageSegmentationHelper = segmentationHelper

        // Run image segmentation on a demo image.
        self.showDemoSegmentation()
      case .failure(_):
        print("Failed to initialize.")
      }
    }
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

  /// Handle tapping on different display mode: Input, Segmentation, Overlay
  @IBAction func onSegmentChanged(_ sender: Any) {
    // `resultImageView` is placed on top of the `inputImageView`. The visibility (alpha value) of
    // `resultImageView` is adjusted depending on the display mode to show/hide/overlay the
    // underlying `inputImageView`.
    switch segmentedControl.selectedSegmentIndex {
    case 0:
      // Mode 0: Show input image
      resultImageView.alpha = 0
    case 1:
      // Mode 1: Show visualization of segmentation result.
      resultImageView.alpha = 1
    case 2:
      // Mode 2: Show overlay of segmentation result on input image.
      resultImageView.alpha = 0.5
    default:
      break
    }
  }

  /// Handle changing center crop setting.
  @IBAction func onCropSwitchValueChanged(_ sender: Any) {
    // Make sure that cached segmentation target image is available.
    guard targetImage != nil else {
      self.inferenceStatusLabel.text = "ERROR: Input image is nil."
      return
    }

    // Re-run the segmentation upon center-crop setting changed.
    runSegmentation(targetImage!)
  }
}

// MARK: - Image Segmentation

extension ViewController {
  /// Run image segmentation on the given image, and show result on screen.
  ///  - Parameter image: The target image for segmentation.
  func runSegmentation(_ image: UIImage) {
    clearResults()

    // Rotate target image to .up orientation to avoid potential orientation misalignment.
    guard let targetImage = image.transformOrientationToUp() else {
      inferenceStatusLabel.text = "ERROR: Image orientation couldn't be fixed."
      return
    }

    // Make sure that image segmentator is initialized.
    guard let imageSegmentator = imageSegmentationHelper else {
      inferenceStatusLabel.text = "ERROR: Image Segmentator is not ready."
      return
    }

    // Cache the target image.
    self.targetImage = targetImage

    // Center-crop the target image if the user has enabled the option.
    let willCenterCrop = cropSwitch.isOn
    guard let inputImage = willCenterCrop ? targetImage.cropCenter() : targetImage else {
      inferenceStatusLabel.text = "ERROR: Image could not be cropped."
      return
    }

    // Cache the potentially cropped image as input to the segmentation model.
    segmentationInput = inputImage

    // Show the potentially cropped image on screen.
    inputImageView.image = inputImage

    // Lock the crop switch while segmentation is running.
    cropSwitch.isEnabled = false

    // Run image segmentation.
    imageSegmentator.runSegmentation(
      inputImage,
      completion: { result in
        // Unlock the crop switch
        self.cropSwitch.isEnabled = true

        // Show the segmentation result on screen
        switch result {
        case let .success(segmentationResult):
          self.segmentationResult = segmentationResult

          // Change to show segmentation overlay result
          self.segmentedControl.selectedSegmentIndex = 2
          self.onSegmentChanged(self)

          // Show result metadata
          self.showInferenceTime(segmentationResult)
          self.showClassLegend(segmentationResult)

          // Show segmentation result
          self.resultImageView.image = segmentationResult.resultImage

          // Enable switching between different display mode: input, segmentation, overlay
          self.segmentedControl.isEnabled = true
        case let .failure(error):
          self.inferenceStatusLabel.text = error.localizedDescription
        }
      })
  }

  /// Clear result from previous run to prepare for new segmentation run.
  private func clearResults() {
    inferenceStatusLabel.text = "Running inference with TensorFlow Lite..."
    legendLabel.text = nil
    segmentedControl.isEnabled = false
    segmentedControl.selectedSegmentIndex = 0
  }

  /// Demo image segmentation with a bundled image.
  private func showDemoSegmentation() {
    if let filePath = Bundle.main.path(forResource: "boy", ofType: "jpg"),
      let image = UIImage(contentsOfFile: filePath)
    {
      runSegmentation(image)
    }
  }

  /// Show segmentation latency on screen.
  private func showInferenceTime(_ segmentationResult: ImageSegmentationResult) {
    let timeString =
      "Model inference: \(Int(segmentationResult.inferenceTime * 1000))ms.\n"
      + "Postprocessing: \(Int(segmentationResult.postProcessingTime * 1000))ms.\n"

    inferenceStatusLabel.text = timeString
  }

  /// Show color legend of each class found in the image.
  private func showClassLegend(_ segmentationResult: ImageSegmentationResult) {
    let legendText = NSMutableAttributedString()

    // Loop through the classes founded in the image.
    segmentationResult.colorLegend.forEach { (className, color) in
      // If the color legend is light, use black text font. If not, use white text font.
      let textColor = color.isLight() ?? true ? UIColor.black : UIColor.white

      // Construct the legend text for current class.
      let attributes = [
        NSAttributedString.Key.font: UIFont.preferredFont(forTextStyle: .headline),
        NSAttributedString.Key.backgroundColor: color,
        NSAttributedString.Key.foregroundColor: textColor,
      ]
      let string = NSAttributedString(string: " \(className) ", attributes: attributes)

      // Add class legend to string to show on the screen.
      legendText.append(string)
      legendText.append(NSAttributedString(string: "  "))
    }

    // Show the class legends on the screen.
    legendLabel.attributedText = legendText
  }
}

// MARK: - UIImagePickerControllerDelegate

extension ViewController: UIImagePickerControllerDelegate, UINavigationControllerDelegate {

  func imagePickerController(
    _ picker: UIImagePickerController,
    didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey: Any]
  ) {

    if let pickedImage = info[.originalImage] as? UIImage {
      runSegmentation(pickedImage)
    }

    dismiss(animated: true)
  }
}
