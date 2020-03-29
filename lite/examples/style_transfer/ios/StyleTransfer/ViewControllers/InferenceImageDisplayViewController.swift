//
//  InferenceImageDisplayViewController.swift
//  StyleTransfer
//
//  Created by Ivan Cheung on 3/28/20.
//  Copyright © 2020 Ivan Cheung. All rights reserved.
//

import UIKit

protocol InferenceImageDisplayViewControllerDelegate: AnyObject {
    func didTapStyleSelectionButton()
    func didTapCameraButton()
}

class InferenceImageDisplayViewController: UIViewController {
    // MARK: Storyboard Outlets
    @IBOutlet private weak var imageView: UIImageView!
    
    weak var delegate: InferenceImageDisplayViewControllerDelegate?
    
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
    
    @IBAction func didTapStyleSelectionButton(_ sender: Any) {
        delegate?.didTapStyleSelectionButton()
    }
    
    @IBAction func didTapCameraButton(_ sender: Any) {
        delegate?.didTapCameraButton()
    }
}
