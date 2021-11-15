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

import Foundation
import Photos
import ios_ImageClassifierBuilderLibObjC

struct ModelTrainer {
  /// Trains a new classification model based on user-selected Photos Library albums. Every image is
  /// labeled after its album name.
  ///
  /// - Parameters:
  ///   - metadata: The metadata of the model to create.
  ///   - albums: The user-selected Photos Library albums.
  /// - Returns: The Model object associated to the newly trained TFLite model.
  func trainModel(metadata: ModelMetadata, on albums: [Album]) -> Model {
    precondition(metadata.isValid)
    precondition(!albums.isEmpty)

    // Copy all images to a reasonable format in tmp.
    let (labels, imagePaths) = prepareImages(from: albums)

    // Create the file URL where the new model will be saved.
    let modelURL = Model.makeURLForNewModel(named: metadata.name)

    // Train the model.
    ModelTrainingUtils.trainModel(
      name: metadata.name, description: metadata.description, author: metadata.author,
      version: metadata.version, license: metadata.license, labels: labels, imagePaths: imagePaths,
      outputModelPath: modelURL.path)

    // Get the number of distinct labels in the labelmap.
    let labelsCount = Classifier(modelURL: modelURL).labels().count
    let size = (try? modelURL.resourceValues(forKeys: [.fileSizeKey]))?.fileSize.map(Int64.init)
    return Model(name: metadata.name, labelsCount: labelsCount, size: size)
  }

  /// Converts albums to a list of labels (the albums names, repeated as many times as there are
  /// images in it) and an equally long list of image paths on disk (typically prepared copies
  /// stored in the tmp directory).
  ///
  /// - Parameter albums: The user-selected Photos Library albums.
  /// - Returns: The pair of lists of labels and image paths on disk.
  private func prepareImages(from albums: [Album]) -> (labels: [String], imagePaths: [String]) {
    var labels: [String] = []
    var imagePaths: [String] = []
    let dispatchGroup = DispatchGroup()
    for album in albums {
      // Iterate over the images.
      let fetchResult = PHAsset.fetchAssets(in: album.collection, options: nil)
      fetchResult.enumerateObjects { (asset, index, stop) in
        guard asset.mediaType == .image else { return }

        // Get the image URL.
        dispatchGroup.enter()
        asset.requestContentEditingInput(with: PHContentEditingInputRequestOptions()) {
          (input, _) in
          defer { dispatchGroup.leave() }
          guard let input = input,
            let imageURLInLibrary = input.fullSizeImageURL,
            let label = album.collection.localizedTitle
          else {
            return
          }

          // Copy the image as resized JPEG to the temporary directory.
          let tmpDirectory = URL(fileURLWithPath: NSTemporaryDirectory())
          let options = PHImageRequestOptions()
          options.isSynchronous = true
          dispatchGroup.enter()
          PHImageManager.default().requestImage(
            for: asset, targetSize: CGSize(width: 640, height: 480), contentMode: .aspectFill,
            options: options
          ) { (image, _) in
            defer { dispatchGroup.leave() }
            if let data = image?.jpegData(compressionQuality: 1) {
              var imageURL =
                tmpDirectory.appendingPathComponent(imageURLInLibrary.lastPathComponent)
              imageURL.deletePathExtension()
              imageURL.appendPathExtension(for: .jpeg)
              do {
                try data.write(to: imageURL, options: .atomic)
              } catch {
                fatalError("Couldn't save training image to \(imageURL): \(error)")
              }
              labels.append(label)
              imagePaths.append(imageURL.path)
            }
          }
        }
      }
    }
    dispatchGroup.wait()
    return (labels, imagePaths)
  }
}
