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
    print(data.score)
    if !data.score.isNaN {
      scoreWidthLayoutConstraint.constant = UIScreen.main.bounds.width/4*CGFloat(data.score)
    } else {
      scoreWidthLayoutConstraint.constant = 0
    }
  }
}
