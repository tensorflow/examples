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
import SwiftUI

/// Shows the live feed from the camera and vends individual frames are accessible via the `onFrame`
/// callback.
struct CameraView: UIViewControllerRepresentable {
  /// The queue on which to call `onFrame`.
  let queue: DispatchQueue
  /// The callback providing individual frames as image buffer.
  ///
  /// This is called on `queue`.
  let onFrame: (CVImageBuffer) -> Void

  func makeUIViewController(context: Context) -> CameraViewController {
    CameraViewController(sampleBufferDelegate: context.coordinator, queue: queue)
  }

  func updateUIViewController(_ uiViewController: CameraViewController, context: Context) {
  }

  func makeCoordinator() -> Coordinator {
    Coordinator(self)
  }

  /// Registers as `CameraViewController` delegate to receive frames, then calls the `CameraView`'s
  /// `onFrame` callback.
  final class Coordinator: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    private let parent: CameraView

    fileprivate init(_ parent: CameraView) {
      self.parent = parent
    }

    func captureOutput(
      _ output: AVCaptureOutput,
      didOutput sampleBuffer: CMSampleBuffer,
      from connection: AVCaptureConnection
    ) {
      if let imageBuffer = sampleBuffer.imageBuffer {
        parent.onFrame(imageBuffer)
      }
    }
  }
}
