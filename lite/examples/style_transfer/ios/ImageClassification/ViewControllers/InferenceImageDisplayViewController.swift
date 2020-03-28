//
//  InferenceImageDisplayViewController.swift
//  ImageClassification
//
//  Created by Ivan Cheung on 3/28/20.
//  Copyright © 2020 Y Media Labs. All rights reserved.
//

import UIKit

class InferenceImageDisplayViewController: UIViewController {
    // MARK: Storyboard Outlets
    @IBOutlet weak var imageView: UIImageView!
    
    // MARK: Instance variables
    var image: UIImage? {
        get { return imageView.image }
        set {
            imageView.image = newValue
        }
    }
    
    // MARK: Constants
    private let bottomSheetButtonDisplayHeight: CGFloat = 100
    
    // MARK: Computed properties
    var collapsedHeight: CGFloat {
      return bottomSheetButtonDisplayHeight
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
    }
}
