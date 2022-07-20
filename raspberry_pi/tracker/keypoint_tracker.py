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
"""Keypoint tracker implementation."""

import math
from typing import List

from data import Person
from tracker.tracker import Track
from tracker.tracker import Tracker


class KeypointTracker(Tracker):
  """KeypointTracker, which tracks poses based on keypoint similarity.

  This tracker assumes that keypoints are provided in normalized image
  coordinates.
  """

  def _compute_similarity(self, persons: List[Person]) -> List[List[float]]:
    """Computes similarity based on Object Keypoint Similarity (OKS).

    Args:
        persons: An array of detected `Person`s.

    Returns:
      A 2D array of shape [num_det, num_tracks] with pairwise similarity scores
      between detections and tracks.
    """
    if (not persons) or (not self._tracks):
      return [[]]

    sim_matrix = []
    for person in persons:
      row = []
      for track in self._tracks:
        row.append(self._object_keypoint_similarity(person, track))
      sim_matrix.append(row)
    return sim_matrix

  def _object_keypoint_similarity(self, person: Person, track: Track) -> float:
    """Computes the Object Keypoint Similarity (OKS) between a person and track.

    This is similar in spirit to the calculation used by COCO keypoint eval:
    https://cocodataset.org/#keypoints-eval
    In this case, OKS is calculated as:
    (1/sum_i d(c_i, c_ti)) * sum_i exp(-d_i^2/(2*a_ti*x_i^2))*d(c_i, c_ti)
    where:
        d(x, y) is an indicator function which only produces 1 if x and y
    exceed a given threshold (i.e. keypointThreshold), otherwise 0.
        c_i is the confidence of keypoint i from the new person
        c_ti is the confidence of keypoint i from the track
        d_i is the Euclidean distance between the person and track keypoint
        a_ti is the area of the track object (the box covering the
        keypoints)
        x_i is a constant that controls falloff in a Gaussian distribution,
    computed as 2*keypointFalloff[i].

    Args:
      person: A `Person`.
      track: A `Track`.

    Returns:
      The OKS score between the person and the track. This number is between 0
      and 1, and larger values indicate more keypoint similarity.
    """
    box_area = self._area(track) + 1e-6
    oks_total = 0
    num_valid_keypoints = 0
    for i in range(len(person.keypoints)):
      person_kpt = person.keypoints[i]
      track_kpt = track.person.keypoints[i]
      if (person_kpt.score <
          self._config.keypoint_tracker_params.keypoint_confidence_threshold or
          track_kpt.score <
          self._config.keypoint_tracker_params.keypoint_confidence_threshold):
        continue

      num_valid_keypoints += 1
      d_squared = ((person_kpt.coordinate.x - track_kpt.coordinate.x)**2 +
                   (person_kpt.coordinate.y - track_kpt.coordinate.y)**2)
      x = 2 * self._config.keypoint_tracker_params.keypoint_falloff[i]
      oks_total += math.exp(-1 * d_squared / (2 * box_area * (x**2)))
    if (num_valid_keypoints <
        self._config.keypoint_tracker_params.min_number_of_keypoints):
      return 0.0

    return oks_total / num_valid_keypoints

  def _area(self, track: Track) -> float:
    """Computes the area of a bounding box that tightly covers keypoints.

    Args:
        track: A 'Track'.

    Returns:
      The area of the object.
    """
    keypoint = list(
        filter(
            lambda kpt: kpt.score > self._config.keypoint_tracker_params.
            keypoint_confidence_threshold, track.person.keypoints))
    x_min = min([1] + [kpt.coordinate.x for kpt in keypoint])
    y_min = min([1] + [kpt.coordinate.y for kpt in keypoint])
    x_max = max([0] + [kpt.coordinate.x for kpt in keypoint])
    y_max = max([0] + [kpt.coordinate.y for kpt in keypoint])

    return (x_max - x_min) * (y_max - y_min)
