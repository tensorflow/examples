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
"""Shared implementation of pose trackers."""

import abc
from typing import List, NamedTuple

from data import Person
from tracker.config import TrackerConfig


class Track(NamedTuple):
  person: Person
  """A person contain keypoint, bounding_box, score and id."""

  last_timestamp: int
  """The last timestamp (in milliseconds) in which a track is recorded."""


class Tracker(object):
  """A stateful tracker for associating detections between frames.

  This is an abstract base class that performs generic mechanics.
  Implementations must inherit from this class.
  """

  def __init__(self, config: TrackerConfig) -> None:
    """Initializes a Tracker."""
    self._tracks = []
    self._config = config
    self._next_track_id = 0

  def apply(self, persons: List[Person], timestamp: int) -> List[Person]:
    """Tracks person instances across frames based on detections.

    Args:
      persons: An array of detected `Person`s.
      timestamp: The timestamp associated with the incoming Persons, in
        microseconds.

    Returns:
      An updated array of `Person`s with tracking id properties.
    """
    self._filter_old_tracks(timestamp)
    sim_matrix = self._compute_similarity(persons)
    self._assign_tracks(persons, sim_matrix, timestamp)
    self._update_tracks()
    return persons

  @abc.abstractmethod
  def _compute_similarity(self, persons: List[Person]) -> List[List[float]]:
    """Computes pairwise similarity scores between detections and tracks.

    Args:
        persons: An array of detected `Person`s.

    Returns:
      A 2D array of shape [num_det, num_tracks] with pairwise similarity scores
      between detections and tracks.
    """
    pass

  def _filter_old_tracks(self, timestamp: int) -> List[Track]:
    """Filters tracks based on their age.

    Args:
      timestamp: The current timestamp in microseconds.

    Returns:
      Filtered list of tracks.
    """
    self._tracks = list(
        filter(
            lambda track: timestamp - track.last_timestamp <= self._config.
            max_age, self._tracks))
    return self._tracks

  def _assign_tracks(self, persons: List[Person], sim_matrix: List[List[float]],
                     timestamp: int) -> None:
    """Performs a greedy optimization to link detections with tracks.

    The `poses` array is updated in place by providing an `id` property. If
    incoming detections are not linked with existing tracks, new tracks will be
    created.

    Args:
      persons: An array of detected `Person`s. It's assumed that persons are
        sorted from most confident to least confident.
      sim_matrix: A 2D array of shape [num_det, num_tracks] with pairwise
        similarity scores between detections and tracks.
      timestamp: The current timestamp in microseconds.
    """
    unmatched_track_indices = list(range(len(sim_matrix[0])))
    detection_indices = list(range(len(persons)))
    unmatched_detection_indices = []

    for detection_index in detection_indices:
      if not unmatched_track_indices:
        unmatched_detection_indices.append(detection_index)
        continue

      # Assign the detection to the track which produces the highest pairwise
      # similarity score, assuming the score exceeds the minimum similarity
      # threshold.
      max_track_index = -1
      max_similarity = -1
      for track_index in unmatched_track_indices:
        similarity = sim_matrix[detection_index][track_index]
        if (similarity >= self._config.min_similarity and
            similarity > max_similarity):
          max_track_index = track_index
          max_similarity = similarity

      if max_track_index >= 0:
        # Link the detection with the highest scoring track.
        linked_track = self._tracks[max_track_index]
        self._tracks[max_track_index] = self._create_track(
            persons[detection_index], timestamp, linked_track.person.id)

        persons[detection_index] = persons[detection_index]._replace(
            id=linked_track.person.id)
        unmatched_track_indices.remove(max_track_index)
      else:
        unmatched_detection_indices.append(detection_index)

    # Spawn new tracks for all unmatched detections.
    for detection_index in unmatched_detection_indices:
      new_track = self._create_track(persons[detection_index], timestamp)
      self._tracks.append(new_track)
      persons[detection_index] = persons[detection_index]._replace(
          id=new_track.person.id)

  def _update_tracks(self) -> None:
    """Updates the stored tracks in the tracker.

    Specifically, the following operations are applied in order:
    * 1. Tracks are sorted based on freshness (i.e. the most recently
    linked
    tracks are placed at the beginning of the array and the most stale
    are at the end).
    * 2. The tracks array is sliced to only contain `maxTracks` tracks
    (i.e. the
    most fresh tracks).

    Returns:
      Updated list of tracks.
    """
    self._tracks = sorted(
        self._tracks, key=lambda track: track.last_timestamp, reverse=True)
    self._tracks = self._tracks[:self._config.max_tracks]

  def _create_track(self,
                    person: Person,
                    timestamp: int,
                    track_id: int = None) -> Track:
    """Creates a track from information in a pose.

    Args:
      person: A `Person`.
      timestamp: The current timestamp in microseconds.
      track_id: The id to assign to the new track. If not provided, will assign
        the next available id.

    Returns:
      A `Track`.
    """
    person = Person(
        person.keypoints, person.bounding_box, person.score,
        track_id if track_id else self._update_and_get_next_track_id())
    track = Track(person, timestamp)
    return track

  def _update_and_get_next_track_id(self):
    """Returns the next track ID."""
    self._next_track_id += 1
    return self._next_track_id

  def _remove(self, ids: List[str]) -> None:
    """Removes specific tracks, based on their ids."""
    self._tracks = list(filter(lambda x: x.person.id not in ids, self._tracks))

  def _reset(self) -> None:
    """Resets tracks."""
    self._tracks = []
