//
//  StylesCollectionViewController.swift
//  StyleTransfer
//
//  Created by Ivan Cheung on 3/28/20.
//  Copyright © 2020 Ivan Cheung. All rights reserved.
//

import UIKit

protocol StylesCollectionViewControllerDelegate: AnyObject {
    func didSelectStyle(style: Style)
}

class StylesCollectionViewController: UICollectionViewController {
    weak var delegate: StylesCollectionViewControllerDelegate?
    var selectedStyle: Style?
    
    // MARK: - Private Properties
    private let reuseIdentifier = "cell"
    private let styles: [Style] = Style.allCases
    
    private let selectedBorderColor: CGColor = UIColor.green.cgColor
    private let selectedBorderWidth: CGFloat = 4
    
    private let itemsPerRow: CGFloat = 3
    private let sectionInsets = UIEdgeInsets(top: 30.0,
                                             left: 20.0,
                                             bottom: 30.0,
                                             right: 20.0)
}

// MARK: - UICollectionViewDataSource
extension StylesCollectionViewController {
    override func numberOfSections(in collectionView: UICollectionView) -> Int {
        return 1
    }
    
    override func collectionView(_ collectionView: UICollectionView,
                                 numberOfItemsInSection section: Int) -> Int {
        return styles.count
    }
    
    override func collectionView(
        _ collectionView: UICollectionView,
        cellForItemAt indexPath: IndexPath
    ) -> UICollectionViewCell {
        let cell = collectionView
            .dequeueReusableCell(withReuseIdentifier: reuseIdentifier, for: indexPath)
        
        if let cell = cell as? ImageCollectionViewCell {
            let style = styles[indexPath.item]
            cell.image = UIImage(named: style.rawValue)
            cell.contentView.layer.borderColor = selectedBorderColor
            cell.contentView.layer.borderWidth = selectedStyle == style ? selectedBorderWidth : 0
        }
        
        return cell
    }
    
    override func collectionView(_ collectionView: UICollectionView, didSelectItemAt indexPath: IndexPath) {
        delegate?.didSelectStyle(style: styles[indexPath.item])
    }
}

// MARK: - Collection View Flow Layout Delegate
extension StylesCollectionViewController : UICollectionViewDelegateFlowLayout {
    
    func collectionView(_ collectionView: UICollectionView,
                        layout collectionViewLayout: UICollectionViewLayout,
                        sizeForItemAt indexPath: IndexPath) -> CGSize {
        let paddingSpace = sectionInsets.left * (itemsPerRow + 1)
        let widthPerItem = (view.frame.width - paddingSpace) / itemsPerRow
        
        return CGSize(width: widthPerItem, height: widthPerItem)
    }
    
    func collectionView(_ collectionView: UICollectionView,
                        layout collectionViewLayout: UICollectionViewLayout,
                        insetForSectionAt section: Int) -> UIEdgeInsets {
        return sectionInsets
    }
    
    func collectionView(_ collectionView: UICollectionView,
                        layout collectionViewLayout: UICollectionViewLayout,
                        minimumLineSpacingForSectionAt section: Int) -> CGFloat {
        return sectionInsets.left
    }
}
