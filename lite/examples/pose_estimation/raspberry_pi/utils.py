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
"""Utility functions to display the pose detection results."""

import cv2
import numpy as np

# Dictionary that maps from joint names to keypoint indices.
KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

# map edges to a RGB color
KEYPOINT_EDGE_INDS_TO_COLOR = {
    (0, 1): (147, 20, 255),
    (0, 2): (255, 255, 0),
    (1, 3): (147, 20, 255),
    (2, 4): (255, 255, 0),
    (0, 5): (147, 20, 255),
    (0, 6): (255, 255, 0),
    (5, 7): (147, 20, 255),
    (7, 9): (147, 20, 255),
    (6, 8): (255, 255, 0),
    (8, 10): (255, 255, 0),
    (5, 6): (0, 255, 255),
    (5, 11): (147, 20, 255),
    (6, 12): (255, 255, 0),
    (11, 12): (0, 255, 255),
    (11, 13): (147, 20, 255),
    (13, 15): (147, 20, 255),
    (12, 14): (255, 255, 0),
    (14, 16): (255, 255, 0)
}


def draw_landmarks_edges(image,
                         keypoint_locs,
                         keypoint_edges,
                         edge_colors,
                         keypoint_color=(0, 255, 0)):
  """Draw landmarks and edges on the input image and return it."""
  for landmark in keypoint_locs:
    landmark_x = min(landmark[0], image.shape[1] - 1)
    landmark_y = min(landmark[1], image.shape[0] - 1)
    cv2.circle(image, (int(landmark_x), int(landmark_y)), 2, keypoint_color, 4)

  for idx, edge in enumerate(keypoint_edges):
    cv2.line(image, (int(edge[0][0]), int(edge[0][1])),
             (int(edge[1][0]), int(edge[1][1])), edge_colors[idx], 2)

  return image


def keypoints_and_edges_for_display(keypoints_with_scores,
                                    height,
                                    width,
                                    keypoint_threshold=0.11):
  """Returns high confidence keypoints and edges for visualization.

  Args:
      keypoints_with_scores: An numpy array with shape [17, 3] representing the
        keypoint coordinates and scores returned by the MoveNet/PoseNet models.
      height: height of the image in pixels.
      width: width of the image in pixels.
      keypoint_threshold: minimum confidence score for a keypoint to be
        visualized.

  Returns:
      A (keypoints_xy, edges_xy, edge_colors) containing:
      * the coordinates of all keypoints of all detected entities;
      * the coordinates of all skeleton edges of all detected entities;
      * the colors in which the edges should be plotted.
  """
  keypoints_all = []
  keypoint_edges_all = []
  edge_colors = []
  kpts_x = keypoints_with_scores[:, 1]
  kpts_y = keypoints_with_scores[:, 0]
  kpts_scores = keypoints_with_scores[:, 2]
  kpts_absolute_xy = np.stack(
      [width * np.array(kpts_x), height * np.array(kpts_y)], axis=-1)
  kpts_above_thresh_absolute = kpts_absolute_xy[
      kpts_scores > keypoint_threshold]
  keypoints_all.append(kpts_above_thresh_absolute)

  for edge_pair, color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
    if (kpts_scores[edge_pair[0]] > keypoint_threshold and
        kpts_scores[edge_pair[1]] > keypoint_threshold):
      x_start = kpts_absolute_xy[edge_pair[0], 0]
      y_start = kpts_absolute_xy[edge_pair[0], 1]
      x_end = kpts_absolute_xy[edge_pair[1], 0]
      y_end = kpts_absolute_xy[edge_pair[1], 1]
      line_seg = np.array([[x_start, y_start], [x_end, y_end]])
      keypoint_edges_all.append(line_seg)
      edge_colors.append(color)
  if keypoints_all:
    keypoints_xy = np.concatenate(keypoints_all, axis=0)
  else:
    num_instances, _ = keypoints_with_scores.shape
    keypoints_xy = np.zeros((0, num_instances, 2))

  if keypoint_edges_all:
    edges_xy = np.stack(keypoint_edges_all, axis=0)
  else:
    edges_xy = np.zeros((0, 2, 2))

  return keypoints_xy, edges_xy, edge_colors
