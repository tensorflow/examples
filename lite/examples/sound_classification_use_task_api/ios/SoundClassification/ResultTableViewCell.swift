//
//  ResultTableViewCell.swift
//  SoundClassification
//
//  Created by MBA0077 on 7/29/22.
//

import UIKit
import TensorFlowLiteTaskAudio

class ResultTableViewCell: UITableViewCell {

  @IBOutlet weak var nameLabel: UILabel!
  @IBOutlet weak var scoreWidthLayoutConstraint: NSLayoutConstraint!

  func setData(_ data: ClassificationCategory) {
    nameLabel.text = data.label
    scoreWidthLayoutConstraint.constant = UIScreen.main.bounds.width/4*CGFloat(data.score)
  }
}
