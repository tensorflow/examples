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

protocol StylePickerViewControllerDelegate {

  func picker(_: StylePickerViewController, didSelectStyle image: UIImage)

}

class StylePickerViewController: UICollectionViewController, UICollectionViewDelegateFlowLayout {

  var delegate: StylePickerViewControllerDelegate?

  class func fromStoryboard() -> StylePickerViewController {
    return UIStoryboard(name: "Main", bundle: nil)
        .instantiateViewController(withIdentifier: "StylePickerViewController")
        as! StylePickerViewController
  }

  private let dataSource = StylePickerDataSource()

  override func viewDidLoad() {
    super.viewDidLoad()
    collectionView.dataSource = dataSource
    collectionView.delegate = self
  }

  override func collectionView(_ collectionView: UICollectionView,
                               didSelectItemAt indexPath: IndexPath) {
    if let image = dataSource.imageForStyle(at: indexPath.item) {
      collectionView.deselectItem(at: indexPath, animated: true)
      delegate?.picker(self, didSelectStyle: image)
      dismiss(animated: true, completion: nil)
    }
  }

  func collectionView(_ collectionView: UICollectionView,
                      layout collectionViewLayout: UICollectionViewLayout,
                      sizeForItemAt indexPath: IndexPath) -> CGSize {
    guard let layout = collectionViewLayout as? UICollectionViewFlowLayout else { return .zero }
    let smallestDimension = collectionView.bounds.width < collectionView.bounds.height ?
        collectionView.bounds.width : collectionView.bounds.height
    let extraPadding: CGFloat = 3 // magic number
    let itemDimension =
        smallestDimension / 2 - collectionView.contentInset.left - collectionView.contentInset.right
        - layout.sectionInset.left - layout.sectionInset.right - layout.minimumInteritemSpacing
        - extraPadding
    return CGSize(width: itemDimension, height: itemDimension)
  }

}

class StylePickerDataSource: NSObject, UICollectionViewDataSource {

  static func defaultStyle() -> UIImage {
    return UIImage(named: "style24")! // use great wave as default
  }

  lazy private var images: [UIImage] = {
    var index = 0
    var images: [UIImage] = []
    while let image = UIImage(named: "style\(index)") {
      images.append(image)
      index += 1
    }
    return images
  }()

  func imageForStyle(at index: Int) -> UIImage? {
    return images[index]
  }

  func collectionView(_ collectionView: UICollectionView,
                      numberOfItemsInSection section: Int) -> Int {
    return images.count
  }

  func collectionView(_ collectionView: UICollectionView,
                      cellForItemAt indexPath: IndexPath) -> UICollectionViewCell {
    let cell = collectionView.dequeueReusableCell(withReuseIdentifier: "StylePickerCell",
                                                  for: indexPath) as! StylePickerCollectionViewCell
    let image = imageForStyle(at: indexPath.item)
    cell.styleImageView.image = image
    return cell
  }

}

class StylePickerCollectionViewCell: UICollectionViewCell {

  @IBOutlet weak var styleImageView: UIImageView!

}
