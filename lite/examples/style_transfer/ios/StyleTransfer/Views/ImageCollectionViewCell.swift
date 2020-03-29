//
//  ImageCollectionViewCell.swift
//  StyleTransfer
//
//  Created by Ivan Cheung on 3/28/20.
//  Copyright © 2020 Ivan Cheung. All rights reserved.
//

import UIKit
class ImageCollectionViewCell: UICollectionViewCell {

    @IBOutlet private weak var imageView: UIImageView!
    
    // MARK: Instance variables
    var image: UIImage? {
        get { return imageView.image }
        set {
            imageView.image = newValue
        }
    }
}
