// Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

import AVFoundation
import CoreVideo
import UIKit

/// Shows the live feed from the camera and vends individual frames are provided to the
/// `sampleBufferDelegate` on `queue`.
class CameraViewController: UIViewController {
  /// The delegate receiving the sample buffers.
  let sampleBufferDelegate: AVCaptureVideoDataOutputSampleBufferDelegate
  /// The queue on which the delegate is receiving the sample buffers.
  let queue: DispatchQueue

  /// The underlying capture session.
  private let captureSession = AVCaptureSession()
  /// The underlying layer displaying the live camera feed.
  private var videoPreviewLayer: AVCaptureVideoPreviewLayer?

  /// Initializes a `CameraViewController` with the details to receive camera frames.
  ///
  /// - Parameters:
  ///   - sampleBufferDelegate: The delegate receiving the sample buffers.
  ///   - queue: The queue on which the delegate is receiving the sample buffers.
  init(sampleBufferDelegate: AVCaptureVideoDataOutputSampleBufferDelegate, queue: DispatchQueue) {
    self.sampleBufferDelegate = sampleBufferDelegate
    self.queue = queue
    super.init(nibName: nil, bundle: nil)
  }

  required init(coder: NSCoder) { fatalError() }

  override func viewDidLoad() {
    super.viewDidLoad()
    view.backgroundColor = .black

    switch AVCaptureDevice.authorizationStatus(for: .video) {
    case .authorized:
      setupCaptureSession()
    case .notDetermined:
      AVCaptureDevice.requestAccess(for: .video) { granted in
        if granted {
          DispatchQueue.main.async { [weak self] in
            self?.setupCaptureSession()
          }
        }
      }
    default:
      print("Camera access not authorized.")
    }
  }

  override func viewWillAppear(_ animated: Bool) {
    super.viewWillAppear(animated)
    captureSession.startRunning()
  }

  override func viewDidDisappear(_ animated: Bool) {
    super.viewDidDisappear(animated)
    captureSession.stopRunning()
  }

  override func viewDidLayoutSubviews() {
    super.viewDidLayoutSubviews()
    if let videoPreviewLayer = videoPreviewLayer {
      videoPreviewLayer.frame = view.bounds
    }
  }

  override func viewWillTransition(
    to size: CGSize,
    with coordinator: UIViewControllerTransitionCoordinator
  ) {
    super.viewWillTransition(to: size, with: coordinator)
    if let connection = videoPreviewLayer?.connection {
      switch UIDevice.current.orientation {
      case .portrait, .unknown, .faceUp, .faceDown:
        connection.videoOrientation = .portrait
      case .portraitUpsideDown:
        connection.videoOrientation = .portraitUpsideDown
      case .landscapeLeft:
        connection.videoOrientation = .landscapeRight
      case .landscapeRight:
        connection.videoOrientation = .landscapeLeft
      @unknown default:
        connection.videoOrientation = .portrait
      }
    }
  }

  /// Configures the capture session inputs and outputs, as well as the layer displaying the live
  /// camera feed.
  private func setupCaptureSession() {
    guard let camera = AVCaptureDevice.default(for: .video) else {
      print("Unable to access camera.")
      return
    }
    do {
      let input = try AVCaptureDeviceInput(device: camera)
      captureSession.addInput(input)

      let output = AVCaptureVideoDataOutput()
      output.videoSettings =
        [kCVPixelBufferPixelFormatTypeKey: Int(kCVPixelFormatType_32BGRA)] as [String: Any]
      output.alwaysDiscardsLateVideoFrames = true
      output.setSampleBufferDelegate(sampleBufferDelegate, queue: queue)
      captureSession.addOutput(output)

      videoPreviewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
      if let videoPreviewLayer = videoPreviewLayer {
        videoPreviewLayer.videoGravity = .resizeAspectFill
        videoPreviewLayer.connection?.videoOrientation = .portrait
        view.layer.addSublayer(videoPreviewLayer)
      }
    } catch let error {
      print("Error Unable to initialize camera:  \(error.localizedDescription)")
    }
  }
}
