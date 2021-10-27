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
"""Bounding box tracker implementation."""

from typing import List

from data import Person
from tracker.tracker import Track
from tracker.tracker import Tracker


class BoundingBoxTracker(Tracker):
  """Tracks objects based on bounding box similarity.

  Similarity is currently defined as intersection-over-union (IoU).
  """

  def _compute_similarity(self, persons: List[Person]) -> List[List[float]]:
    """Computes similarity based on intersection-over-union (IoU).

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
        row.append(self._iou(person, track))
      sim_matrix.append(row)
    return sim_matrix

  def _iou(self, person: Person, track: Track) -> float:
    """Computes the intersection-over-union (IoU) between a pose and a track.

    Args:
      person: A `Person`.
      track: A `Track`.

    Returns:
      The IoU  between the person and the track. This number is between 0 and 1,
      and larger values indicate more box similarity.
    """
    x_min = max(person.bounding_box.start_point.x,
                track.person.bounding_box.start_point.x)
    y_min = max(person.bounding_box.start_point.y,
                track.person.bounding_box.start_point.y)
    x_max = min(person.bounding_box.end_point.x,
                track.person.bounding_box.end_point.x)
    y_max = min(person.bounding_box.end_point.y,
                track.person.bounding_box.end_point.y)

    person_width = person.bounding_box.end_point.x - person.bounding_box.start_point.x
    person_height = person.bounding_box.end_point.y - person.bounding_box.start_point.y
    track_width = track.person.bounding_box.end_point.x - track.person.bounding_box.start_point.x
    track_height = track.person.bounding_box.end_point.y - track.person.bounding_box.start_point.y

    if (x_min >= x_max or y_min >= y_max):
      return 0.0

    intersection = (x_max - x_min) * (y_max - y_min)
    area_person = person_width * person_height
    area_track = track_width * track_height
    return float(intersection) / (area_person + area_track - intersection)
