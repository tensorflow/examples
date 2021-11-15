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

import Photos
import SwiftUI

/// Displays the list of Photos Library albums.
///
/// The user can then make a selection before proceeding to the next step.
///
/// This screen is part of the model creation UX.
struct AlbumSelector: View {
  /// The metadata of the model to create.
  let modelMetadata: ModelMetadata
  /// The closure to call once the Model object will be created.
  let completion: (Model?) -> Void
  /// The list of albums from the Photos Library.
  @State private var albums: [Album] = []

  var body: some View {
    VStack {
      if PHPhotoLibrary.authorizationStatus(for: .readWrite) == .authorized {
        Text(
          "The model classes will be named after the Photos Library albums you select. Make sure "
            + "an album contains images of only one class."
        )
        .padding()
        List {
          Section(header: Text("Albums")) {
            ForEach(albums.indices, id: \.self) { index in
              Row(album: albums[index])
                .onTapGesture {
                  albums[index].selected.toggle()
                }
            }
          }
        }
        .listStyle(GroupedListStyle())
      } else {
        Text(
          "Classification by Retrieval lets you select albums from your library to train the "
            + "model. Please allow access in Settings."
        )
        .padding()
        Button("Settings") {
          UIApplication.shared.open(URL(string: UIApplication.openSettingsURLString)!)
        }
      }
    }
    .onAppear {
      fetchAlbums()
    }
    .navigationBarTitle(Text("Select Albums"), displayMode: .inline)
    .toolbar {
      ToolbarItem(placement: .primaryAction) {
        NavigationLink(
          "Next",
          destination: ModelTrainerView(
            modelMetadata: modelMetadata, albums: albums.filter(\.selected), completion: completion)
        )
        .accessibilityHint(
          "Proceeds to the next model creation step. Dimmed until at least two albums are selected."
        )
        .disabled(albums.filter(\.selected).count < 2)
      }
    }
  }

  /// Requests authorization to access the library and fetches the albums if authorized.
  func fetchAlbums() {
    PHPhotoLibrary.requestAuthorization(for: .readWrite) { status in
      switch status {
      case .notDetermined:
        print("Photo Library authorization not determined.")
      case .restricted:
        print("Photo Library authorization restricted.")
      case .denied:
        print("Photo Library authorization denied.")
      case .authorized:
        self.albums = PHCollectionList.fetchAlbums()
      case .limited:
        print("Photo Library authorization limited.")
      @unknown default:
        fatalError()
      }
    }
  }

  /// Displays an album in the list.
  struct Row: View {
    /// The represented album.
    let album: Album

    var body: some View {
      HStack {
        Image(systemName: "photo")
        HStack(alignment: .lastTextBaseline) {
          Text(album.collection.localizedTitle!)
          Text(
            "\(album.collection.estimatedAssetCount) "
              + "\(album.collection.estimatedAssetCount > 1 ? "photos" : "photo")"
          )
          .font(.caption)
          .foregroundColor(.secondary)
        }
        Spacer()
        Image(systemName: album.selected ? "checkmark.circle.fill" : "circle")
      }
      .contentShape(Rectangle())
      .accessibilityAddTraits(album.selected ? [.isSelected] : [])
      .accessibilityElement(children: .combine)
    }
  }
}

extension PHCollectionList {
  /// Fetches the Photos Library albums.
  static func fetchAlbums() -> [Album] {
    var albums = [Album]()
    let userCollections = fetchTopLevelUserCollections(with: nil)
    userCollections.enumerateObjects { (collection, index, stop) in
      if let collection = collection as? PHAssetCollection, collection.localizedTitle != nil {
        albums.append(Album(collection: collection))
      }
    }
    return albums
  }
}
