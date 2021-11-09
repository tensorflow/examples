# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit test of pose estimation using MoveNet."""

import logging
import unittest

import cv2
from data import BodyPart
from ml.movenet import Movenet
import numpy as np
import pandas as pd

_MODEL_LIGHTNING = 'movenet_lightning'
_MODEL_THUNDER = 'movenet_thunder'
_IMAGE_TEST1 = 'test_data/image1.png'
_IMAGE_TEST2 = 'test_data/image2.jpeg'
_GROUND_TRUTH_CSV = 'test_data/pose_landmark_truth.csv'
_ALLOWED_DISTANCE = 21


class MovenetTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.image_1 = cv2.imread(_IMAGE_TEST1)
    self.image_2 = cv2.imread(_IMAGE_TEST2)

    # Initialize model
    self.movenet_lightning = Movenet(_MODEL_LIGHTNING)
    self.movenet_thunder = Movenet(_MODEL_THUNDER)
    # Get pose landmarks truth
    pose_landmarks_truth = pd.read_csv(_GROUND_TRUTH_CSV)
    self.keypoints_truth_1 = pose_landmarks_truth.iloc[0].to_numpy().reshape(
        (17, 2))
    self.keypoints_truth_2 = pose_landmarks_truth.iloc[1].to_numpy().reshape(
        (17, 2))

  def _detect_and_assert(self, detector: Movenet, image: np.ndarray,
                         keypoints_truth: np.ndarray) -> None:
    """Run pose estimation and assert if the result is close to ground truth.

    Args:
      detector: A Movenet pose estimator.
      image: A [height, width, 3] RGB image.
      keypoints_truth: Ground truth keypoint coordinates to be compared to.
    """
    person = detector.detect(image, reset_crop_region=True)
    keypoints = person.keypoints

    for idx in range(len(BodyPart)):
      kpt_estimate = np.array(
          [keypoints[idx].coordinate.x, keypoints[idx].coordinate.y])
      distance = np.linalg.norm(kpt_estimate - keypoints_truth[idx], np.inf)

      self.assertGreaterEqual(
          _ALLOWED_DISTANCE, distance,
          '{0} is too far away ({1}) from ground truth data.'.format(
              BodyPart(idx).name, int(distance)))
      logging.debug('Detected %s close to expected result (%d)',
                    BodyPart(idx).name, int(distance))

  def test_pose_estimation_image1_lightning(self):
    """Test if MoveNet Lightning detection's close to ground truth of image1."""
    self._detect_and_assert(self.movenet_lightning, self.image_1,
                            self.keypoints_truth_1)

  def test_pose_estimation_image1_thunder(self):
    """Test if MoveNet Thunder detection's close to ground truth of image1."""
    self._detect_and_assert(self.movenet_thunder, self.image_1,
                            self.keypoints_truth_1)

  def test_pose_estimation_image2_lightning(self):
    """Test if MoveNet Lightning detection's close to ground truth of image2."""
    self._detect_and_assert(self.movenet_lightning, self.image_2,
                            self.keypoints_truth_2)

  def test_pose_estimation_image2_thunder(self):
    """Test if MoveNet Thunder detection's close to ground truth of image2."""
    self._detect_and_assert(self.movenet_thunder, self.image_2,
                            self.keypoints_truth_2)


if __name__ == '__main__':
  unittest.main()
