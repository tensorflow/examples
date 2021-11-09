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
"""Unit test of pose estimation using MoveNet Multipose."""

import logging
from typing import List
import unittest

import cv2
from data import BodyPart
from data import KeyPoint
from .movenet_multipose import MoveNetMultiPose
import numpy as np
import pandas as pd

_MODEL_MOVENET_MULTILPOSE = 'movenet_multipose'
_IMAGE_TEST1 = 'test_data/image1.png'
_IMAGE_TEST2 = 'test_data/image2.jpeg'
_GROUND_TRUTH_CSV = 'test_data/pose_landmark_truth.csv'
_ALLOWED_DISTANCE = 41


class MovenetMultiPoseTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    image_1 = cv2.imread(_IMAGE_TEST1)
    image_2 = cv2.imread(_IMAGE_TEST2)

    # Merge image_1 and image_2 into a single image for testing MultiPose model.
    image = cv2.hconcat([image_1, image_2])

    # Initialize the MultiPose model.
    detector = MoveNetMultiPose(_MODEL_MOVENET_MULTILPOSE)

    # Run detection on the merged image
    self.list_persons = detector.detect(image)

    # Sort the results so that the person on the right side come first.
    self.list_persons.sort(key=lambda person: person.bounding_box.start_point.x)

    # Load the pose landmarks ground truth.
    pose_landmarks_truth = pd.read_csv(_GROUND_TRUTH_CSV)
    keypoints_truth_1 = pose_landmarks_truth.iloc[0].to_numpy().reshape((17, 2))
    keypoints_truth_2 = pose_landmarks_truth.iloc[1].to_numpy().reshape((17, 2))

    # Shift keypoints_truth_2 to the right to account for the space occupied by
    # image1.
    for idx in range(keypoints_truth_2.shape[0]):
      keypoints_truth_2[idx][0] += image_1.shape[1]

    self.keypoints_truth = [keypoints_truth_1, keypoints_truth_2]

  def _assert(self, keypoints: List[KeyPoint],
              keypoints_truth: np.ndarray) -> None:
    """Assert if the detection result is close to ground truth.

    Args:
      keypoints: List Keypoint detected by from the Movenet Multipose model.
      keypoints_truth: Ground truth keypoints.
    """
    for idx in range(len(BodyPart)):
      kpt_estimate = np.array(
          [keypoints[idx].coordinate.x, keypoints[idx].coordinate.y])
      kpt_truth = keypoints_truth[idx]
      distance = np.linalg.norm(kpt_estimate - kpt_truth, np.inf)

      self.assertGreaterEqual(
          _ALLOWED_DISTANCE, distance,
          '{0} is too far away ({1}) from ground truth data.'.format(
              BodyPart(idx).name, int(distance)))
      logging.debug('Detected %s close to expected result (%d)',
                    BodyPart(idx).name, int(distance))

  def test_pose_estimation_image1_multipose(self):
    """Test if MoveNet Multipose's detection is close to ground truth of image1."""
    keypoints = self.list_persons[0].keypoints
    self._assert(keypoints, self.keypoints_truth[0])

  def test_pose_estimation_image2_multipose(self):
    """Test if MoveNet Multipose's detection is close to ground truth of image2."""
    keypoints = self.list_persons[1].keypoints
    self._assert(keypoints, self.keypoints_truth[1])


if __name__ == '__main__':
  unittest.main()
