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
"""Configuration of pose trackers."""
from typing import NamedTuple, List


class KeypointTrackerConfig(NamedTuple):
  """Keypoint tracker specific configuration."""

  keypoint_confidence_threshold: float = 0.3
  """The minimum keypoint confidence threshold.

    A keypoint is only compared in the OKS calculation if both the new detected
    keypoint and the
    corresponding track keypoint have confidences above this threshold
  """

  keypoint_falloff: List[float] = [
      0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062,
      0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089
  ]
  """Per-keypoint falloff in OKS calculation."""

  min_number_of_keypoints: int = 4
  """The minimum number of keypoints that are necessary for computing OKS.

    If the number of confident keypoints (between a pose and track) are under
    this value, an OKS of 0.0
    will be given.
  """


class TrackerConfig(NamedTuple):
  """Shared config for pose trackers."""

  keypoint_tracker_params: KeypointTrackerConfig = KeypointTrackerConfig()
  """Keypoint tracker params."""

  max_tracks: int = 4
  """The maximum number of tracks that an internal tracker will maintain."""

  max_age: int = 1000 * 1000
  """ Maximum track lifetime.

    The maximum duration of time (in milliseconds) that a track can exist
    without being
    linked with a new detection before it is removed. Set this value large if
    you would
    like to recover people that are not detected for long stretches of time (at
    the cost
    of potential false re-identifications).
  """

  min_similarity: float = 0.4
  """New poses will only be linked with tracks if the similarity score exceeds this threshold."""
