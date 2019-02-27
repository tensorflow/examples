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

  // MARK: Storyboard Outlets
  @IBOutlet weak var collectionView: UICollectionView!

  // MARK: Constants
  private let unselectedFontColor = UIColor(
      displayP3Red: 124.0/255.0, green: 136.0/255.0, blue: 144.0/255.0, alpha: 1.0)
  private let selectedFontColor = UIColor(
      displayP3Red: 250.0/255.0, green: 141.0/255.0, blue: 0.0/255.0, alpha: 1.0)
  private let unselectedBorderColor = UIColor(
      displayP3Red: 199.0/255.0, green: 208.0/255.0, blue: 216.0/255.0, alpha: 1.0)
  private let collectionViewPadding: CGFloat = 15.0
  private let highlightTime: Double = 0.5
  private let imageInset: CGFloat = 8.0

  // MARK: Objects Handling Core Functionality
  private let modelDataHandler = ModelDataHandler(
      modelFileName: "conv_actions_frozen", labelsFileName: "conv_actions_labels",
      labelsFileExtension: "txt")
  private var audioInputManager: AudioInputManager?
  private var inferenceViewController: InferenceViewController?

  // MARK: Instance Variables
  private var words: [String] = []
  private var result: Result?
  private var highlightedCommand: RecognizedCommand?
  private var bufferSize: Int = 0

  // MARK: View Handling Methods
  override func viewDidLoad() {
    super.viewDidLoad()

    guard let handler = modelDataHandler else {
      return
    }

    // Displays lables
    words = handler.offSetLabelsForDisplay()
    self.collectionView.reloadData()

    startAudioRecognition()

  }

  override var preferredStatusBarStyle : UIStatusBarStyle {
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
      inferenceViewController?.sampleRate = Int(tempModelDataHandler.sampleRate)
      inferenceViewController?.threadCountLimit = tempModelDataHandler.threaCountLimit
      inferenceViewController?.currentThreadCount = tempModelDataHandler.threadCount
      inferenceViewController?.delegate = self
    }
  }

  /**
   Initializes the AudioInputManager and starts recognizing on the output buffers.
   */
  private func startAudioRecognition() {

    guard let handler = modelDataHandler else {
      return
    }

    audioInputManager = AudioInputManager(sampleRate: handler.sampleRate)
    audioInputManager?.delegate = self

    guard let workingAudioInputManager = audioInputManager else {
      return
    }

    bufferSize = workingAudioInputManager.bufferSize

    workingAudioInputManager.checkPermissionsAndStartTappingMicrophone()
//    workingAudioInputManager.start { (channelDataArray) in
//
//      self.runModel(onBuffer: Array(channelDataArray[0..<handler.sampleRate]))
//      self.runModel(onBuffer: Array(channelDataArray[handler.sampleRate..<bufferSize]))
//    }
  }

  /**
   This method runs hands off inference to the ModelDataHandler by passing the audio buffer.
   */
  private func runModel(onBuffer buffer: [Int16]) {

    result = modelDataHandler?.runModel(onBuffer: buffer)

    // Updates the results on the screen.
    DispatchQueue.main.async {
      self.refreshInferenceTime()
      guard let recognizedCommand = self.result?.recognizedCommand else {
        return
      }
      self.highlightedCommand =  recognizedCommand
      self.highlightResult()
    }
  }

  /**
   Highlights the recognized command in the UICollectionView for the specified time.
   */
  private func highlightResult() {

    DispatchQueue.main.async {

      self.collectionView.reloadData()
      self.perform(#selector(ViewController.unhighlightResult), with: nil, afterDelay: self.highlightTime)
    }
  }

  /**
   Unhighlights the recognized command in the UICollectionView.
   */
  @objc func unhighlightResult() {
    highlightedCommand = nil

    collectionView.reloadData()
  }

  /**
   Refreshes the additional information displayed by InferenceViewController.
   */
  func refreshInferenceTime() {

    var inferenceTime: Double = 0.0
    if let result = self.result {
      inferenceTime = result.inferenceTime
    }
    inferenceViewController?.inferenceTime = inferenceTime
    inferenceViewController?.refreshResults()
  }
}

// MARK: InferenceViewControllerDelegate Methods
extension ViewController: InferenceViewControllerDelegate {
  func didChangeThreadCount(to count: Int) {
    modelDataHandler?.set(numberOfThreads: count)
  }
}

// MARK: UICollectionView DataSource and Delegate
extension ViewController: UICollectionViewDelegate, UICollectionViewDataSource, UICollectionViewDelegateFlowLayout {

  // Get item size of the collection view with respect to it's current width and height.
  private func itemSize() -> CGSize {
    let width = (self.collectionView.bounds.size.width - collectionViewPadding) / 2.0
    let rows: CGFloat = CGFloat(words.count / 2)
    let height =  (self.collectionView.bounds.size.height - ((CGFloat(rows - 1) * collectionViewPadding))) /  rows

    return CGSize(width: width, height: height)
  }

  func numberOfSections(in collectionView: UICollectionView) -> Int {
    return 1
  }

  func collectionView(_ collectionView: UICollectionView, numberOfItemsInSection section: Int) -> Int {
    return words.count
  }

  func collectionView(_ collectionView: UICollectionView, willDisplay cell: UICollectionViewCell, forItemAt indexPath: IndexPath) {

    var borderColor = UIColor.clear
    let wordCell = cell as? WordCell

    let word = words[indexPath.item]

    if let recognizedCommand = highlightedCommand, recognizedCommand.name == word {
      borderColor = UIColor.clear
    }
    else {
      borderColor = unselectedBorderColor
    }

    wordCell?.borderColor = borderColor
    wordCell?.setNeedsDisplay()
  }

  func collectionView(_ collectionView: UICollectionView, layout collectionViewLayout: UICollectionViewLayout, sizeForItemAt indexPath: IndexPath) -> CGSize {

    return itemSize()
  }

  func collectionView(_ collectionView: UICollectionView, cellForItemAt indexPath: IndexPath) -> UICollectionViewCell {

    let cell = collectionView.dequeueReusableCell(withReuseIdentifier: "WORD_CELL", for: indexPath) as! WordCell

    let word = words[indexPath.item]

    var backgroundImage: UIImage?
    var fontColor = unselectedFontColor
    var name = word.capitalized

    if let recognizedCommand = highlightedCommand, recognizedCommand.name == word {
      backgroundImage = UIImage(named: "base")?.resizableImage(withCapInsets: UIEdgeInsetsMake(imageInset, imageInset, imageInset, imageInset), resizingMode: .stretch)
      fontColor = selectedFontColor
      name = word.capitalized + " (\(Int(recognizedCommand.score * 100.0))%)"
    }

    cell.backgroundImageView.image = backgroundImage
    cell.nameLabel.textColor = fontColor
    cell.nameLabel.text = name

    return cell
  }

}

extension ViewController: AudioInputManagerDelegate {

  func didOutput(channelData: [Int16]) {

    guard let handler = modelDataHandler else {
      return
    }

    self.runModel(onBuffer: Array(channelData[0..<handler.sampleRate]))
    self.runModel(onBuffer: Array(channelData[handler.sampleRate..<bufferSize]))
  }

  func showCameraPermissionsDeniedAlert() {

    let alertController = UIAlertController(title: "Microphone Permissions Denied", message: "Microphone permissions have been denied for this app. You can change this by going to Settings", preferredStyle: .alert)

    let cancelAction = UIAlertAction(title: "Cancel", style: .cancel, handler: nil)
    let settingsAction = UIAlertAction(title: "Settings", style: .default) { (action) in
      UIApplication.shared.open(URL(string: UIApplicationOpenSettingsURLString)!, options: [:], completionHandler: nil)
    }

    alertController.addAction(cancelAction)
    alertController.addAction(settingsAction)

    present(alertController, animated: true, completion: nil)
  }
}

