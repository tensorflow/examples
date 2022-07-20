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
"""Script to run visualize pose estimation on test data."""
import argparse
import logging

import cv2
from data import BodyPart
from data import person_from_keypoints_with_scores
from ml import Movenet
from ml import Posenet
import numpy as np
import pandas as pd
import utils

_MODEL_POSENET = 'posenet'
_MODEL_LIGHTNING = 'movenet_lightning'
_MODEL_THUNDER = 'movenet_thunder'
_GROUND_TRUTH_CSV = 'test_data/pose_landmark_truth.csv'
_TEST_IMAGE_PATHS = ['test_data/image1.png', 'test_data/image2.jpeg']

# Load test images
_TEST_IMAGES = [cv2.imread(path) for path in _TEST_IMAGE_PATHS]

# Load pose estimation models
_POSENET = Posenet(_MODEL_POSENET)
_MOVENET_LIGHTNING = Movenet(_MODEL_LIGHTNING)
_MOVENET_THUNDER = Movenet(_MODEL_THUNDER)

# Load pose landmarks truth
_POSE_LANDMARKS_TRUTH = pd.read_csv(_GROUND_TRUTH_CSV)
_KEYPOINTS_TRUTH_LIST = [
    row.to_numpy().reshape((17, 2)) for row in _POSE_LANDMARKS_TRUTH.iloc
]


def _visualize_detection_result(input_image, ground_truth):
  """Visualize the pose estimation result and write the output image to a file.

  The detected keypoints follow these color codes:
      * PoseNet: blue
      * MoveNet Lightning: red
      * MoveNet Thunder: yellow
      * Ground truth (from CSV): green
  Note: This test is meant to be run by a human who want to visually verify
  the pose estimation result.

  Args:
    input_image: Numpy array of shape (height, width, 3)
    ground_truth: Numpy array with absolute coordinates of the keypoints to be
      plotted.

  Returns:
    Input image with pose estimation results.
  """
  output_image = input_image.copy()

  # Draw detection result from Posenet (blue)
  person = _POSENET.detect(input_image)
  output_image = utils.visualize(output_image, [person], (255, 0, 0))

  # Draw detection result from Movenet Lightning (red)
  person = _MOVENET_LIGHTNING.detect(input_image, reset_crop_region=True)
  output_image = utils.visualize(output_image, [person], (0, 0, 255))

  # Draw detection result from Movenet Thunder (yellow)
  person = _MOVENET_THUNDER.detect(input_image, reset_crop_region=True)
  output_image = utils.visualize(output_image, [person], (0, 255, 255))

  # Create a fake score column to convert ground truth to "Person" type
  ground_truth[:, :2] = ground_truth[:, 1::-1]
  score = np.ones((17, 1), dtype=float)
  ground_truth = np.append(ground_truth, score, axis=1)
  person = person_from_keypoints_with_scores(ground_truth, 1, 1)

  # Draw ground truth detection result (green)
  output_image = utils.visualize(output_image, [person], (0, 255, 0))

  return output_image


def _create_ground_truth_csv(input_images, ground_truth_csv_path):
  """Create ground truth CSV file from the given input images.

  Args:
    input_images: An array of input RGB images (height, width, 3).
    ground_truth_csv_path: path to the output CSV.
  """
  # Create column name for CSV file
  column_names = []
  for body_part in BodyPart:
    column_names.append(body_part.name + '_x')
    column_names.append(body_part.name + '_y')

  # Create ground truth data by feeding the test images through MoveNet
  # Thunder 3 times to leverage the cropping logic and improve accuracy.
  keypoints_data = []
  for input_image in input_images:
    person = _MOVENET_THUNDER.detect(input_image, reset_crop_region=True)
    for _ in range(3):
      person = _MOVENET_THUNDER.detect(input_image, reset_crop_region=False)

    kpts = []
    keypoints = person.keypoints
    for idx in range(len(keypoints)):
      kpts.extend((keypoints[idx].coordinate.x, keypoints[idx].coordinate.y))

    # Store kpts into keypoints_data
    keypoints_data.append(kpts)

  # Write ground truth CSV file
  keypoints_df = pd.DataFrame(keypoints_data, columns=column_names)
  keypoints_df.to_csv(ground_truth_csv_path, index=False)


def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--ground_truth_csv_output',
      help='Path to generate ground truth CSV file. (Optional)',
      required=False)
  args = parser.parse_args()

  # Create ground truth CSV if the ground_truth_csv parameter is set
  if args.ground_truth_csv_output:
    _create_ground_truth_csv(_TEST_IMAGES, args.ground_truth_csv_output)
    logging.info('Created ground truth keypoint CSV: %s',
                 args.ground_truth_csv_output)

  # Visualize detection result of the test images
  for index in range(len(_TEST_IMAGES)):
    test_image_path = _TEST_IMAGE_PATHS[index]
    test_image = _TEST_IMAGES[index]
    keypoint_truth = _KEYPOINTS_TRUTH_LIST[index]
    visualized_image = _visualize_detection_result(test_image, keypoint_truth)
    cv2.imshow(test_image_path, visualized_image)

  cv2.waitKey(0)
  cv2.destroyAllWindows()


if __name__ == '__main__':
  main()
