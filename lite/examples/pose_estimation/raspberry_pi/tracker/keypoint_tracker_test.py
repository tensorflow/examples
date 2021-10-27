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
"""Unit test of Keypoint tracker."""

import math
import unittest

from data import BodyPart
from data import KeyPoint
from data import Person
from data import Point
from data import Rectangle
from tracker.config import KeypointTrackerConfig
from tracker.config import TrackerConfig
from tracker.keypoint_tracker import KeypointTracker
from tracker.tracker import Track

_KEYPOINT_CONFIDENCE_THRESHOLD = 0.2
_KEYPOINT_FALLOFF = [0.1, 0.1, 0.1, 0.1]
_MIN_NUMBER_OF_KEYPOINTS = 2


class KeypointTrackerTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    kpt_config = KeypointTrackerConfig(_KEYPOINT_CONFIDENCE_THRESHOLD,
                                       _KEYPOINT_FALLOFF,
                                       _MIN_NUMBER_OF_KEYPOINTS)
    self.tracker_config = TrackerConfig(kpt_config)
    self.kpt_tracker = KeypointTracker(self.tracker_config)

  def test_oks(self):
    """Test OKS."""
    person = Person([
        KeyPoint(BodyPart(0), Point(0.2, 0.2), 1),
        KeyPoint(BodyPart(1), Point(0.4, 0.4), 0.8),
        KeyPoint(BodyPart(2), Point(0.6, 0.6), 0.1),
        KeyPoint(BodyPart(3), Point(0.8, 0.7), 0.8)
    ], Rectangle(Point(0, 0), Point(0, 0)), 1)
    track = Track(
        Person([
            KeyPoint(BodyPart(0), Point(0.2, 0.2), 1),
            KeyPoint(BodyPart(1), Point(0.4, 0.4), 0.8),
            KeyPoint(BodyPart(2), Point(0.6, 0.6), 0.9),
            KeyPoint(BodyPart(3), Point(0.8, 0.8), 0.8)
        ], Rectangle(Point(0, 0), Point(0, 0)), 1), 1000000)

    oks = self.kpt_tracker._object_keypoint_similarity(person, track)

    box_area = (0.8 - 0.2) * (0.8 - 0.2)
    x = 2 * self.tracker_config.keypoint_tracker_params.keypoint_falloff[3]
    d = 0.1
    expected_oks = (1 + 1 + math.exp(-1 * (d**2) / (2 * box_area * (x**2)))) / 3

    self.assertAlmostEqual(oks, expected_oks, 6)

  def test_oks_returns_zero(self):
    """Compute OKS returns 0.0 with less than 2 valid keypoints."""
    person = Person([
        KeyPoint(BodyPart(0), Point(0.2, 0.2), 1),
        KeyPoint(BodyPart(1), Point(0.4, 0.4), 0.1),
        KeyPoint(BodyPart(2), Point(0.6, 0.6), 0.9),
        KeyPoint(BodyPart(3), Point(0.8, 0.8), 0.8)
    ], Rectangle(Point(0, 0), Point(0, 0)), 1)
    track = Track(
        Person([
            KeyPoint(BodyPart(0), Point(0.2, 0.2), 1),
            KeyPoint(BodyPart(1), Point(0.4, 0.4), 0.8),
            KeyPoint(BodyPart(2), Point(0.6, 0.6), 0.1),
            KeyPoint(BodyPart(3), Point(0.8, 0.8), 0.1)
        ], Rectangle(Point(0, 0), Point(0, 0)), 1), 1000000)

    oks = self.kpt_tracker._object_keypoint_similarity(person, track)
    self.assertAlmostEqual(oks, 0.0, 6)

  def test_area(self):
    """Test area."""
    track = Track(
        Person([
            KeyPoint(BodyPart(0), Point(0.1, 0.2), 1),
            KeyPoint(BodyPart(1), Point(0.3, 0.4), 0.9),
            KeyPoint(BodyPart(2), Point(0.4, 0.6), 0.9),
            KeyPoint(BodyPart(3), Point(0.7, 0.8), 0.1)
        ], Rectangle(Point(0, 0), Point(0, 0)), 1), 1000000)

    area = self.kpt_tracker._area(track)
    expected_area = (0.4 - 0.1) * (0.6 - 0.2)
    self.assertAlmostEqual(area, expected_area, 6)

  def test_keypoint_tracker(self):
    """Test Keypoint tracker."""

    # Timestamp: 0. Person becomes the only track
    persons = [
        Person([
            KeyPoint(BodyPart(0), Point(0.2, 0.2), 1),
            KeyPoint(BodyPart(1), Point(0.4, 0.4), 0.8),
            KeyPoint(BodyPart(2), Point(0.6, 0.6), 0.9),
            KeyPoint(BodyPart(3), Point(0.8, 0.8), 0.0)
        ], Rectangle(Point(0, 0), Point(0, 0)), 1)
    ]

    persons = self.kpt_tracker.apply(persons, 0)
    tracks = self.kpt_tracker._tracks
    self.assertEqual(len(persons), 1)
    self.assertEqual(persons[0].id, 1)
    self.assertEqual(len(tracks), 1)
    self.assertEqual(tracks[0].person.id, 1)
    self.assertEqual(tracks[0].last_timestamp, 0)

    # Timestamp: 100000. First person is linked with track 1. Second person
    # spawns a new track (id = 2).
    persons = [
        Person([
            KeyPoint(BodyPart(0), Point(0.2, 0.2), 1),
            KeyPoint(BodyPart(1), Point(0.4, 0.4), 0.8),
            KeyPoint(BodyPart(2), Point(0.6, 0.6), 0.9),
            KeyPoint(BodyPart(3), Point(0.8, 0.8), 0.8)
        ], Rectangle(Point(0, 0), Point(0, 0)), 1),
        Person(
            [
                KeyPoint(BodyPart(0), Point(0.8, 0.8), 0.8),
                KeyPoint(BodyPart(1), Point(0.6, 0.6), 0.3),
                KeyPoint(BodyPart(2), Point(0.4, 0.4), 0.1),  # Low confidence.
                KeyPoint(BodyPart(3), Point(0.2, 0.2), 0.8)
            ],
            Rectangle(Point(0, 0), Point(0, 0)),
            1)
    ]

    persons = self.kpt_tracker.apply(persons, 100000)
    tracks = self.kpt_tracker._tracks
    self.assertEqual(len(persons), 2)
    self.assertEqual(persons[0].id, 1)
    self.assertEqual(persons[1].id, 2)
    self.assertEqual(len(tracks), 2)
    self.assertEqual(tracks[0].person.id, 1)
    self.assertEqual(tracks[0].last_timestamp, 100000)
    self.assertEqual(tracks[1].person.id, 2)
    self.assertEqual(tracks[1].last_timestamp, 100000)

    # Timestamp: 900000. First person is linked with track 2. Second person
    # spawns a new track (id = 3).
    persons = [  # Links with id = 2.
        Person(
            [
                KeyPoint(BodyPart(0), Point(0.6, 0.7), 0.7),
                KeyPoint(BodyPart(1), Point(0.5, 0.6), 0.7),
                KeyPoint(BodyPart(2), Point(0.0, 0.0), 0.1),  # Low confidence.
                KeyPoint(BodyPart(3), Point(0.2, 0.1), 1.0)
            ],
            Rectangle(Point(0, 0), Point(0, 0)),
            1),
        # Becomes id = 3.
        Person(
            [
                KeyPoint(BodyPart(0), Point(0.5, 0.1), 0.6),
                KeyPoint(BodyPart(1), Point(0.9, 0.3), 0.6),
                KeyPoint(BodyPart(2), Point(0.1, 0.1), 0.9),
                KeyPoint(BodyPart(3), Point(0.4, 0.4), 0.1)
            ],  # Low confidence.
            Rectangle(Point(0, 0), Point(0, 0)),
            1)
    ]

    persons = self.kpt_tracker.apply(persons, 900000)
    tracks = self.kpt_tracker._tracks
    self.assertEqual(len(persons), 2)
    self.assertEqual(persons[0].id, 2)
    self.assertEqual(persons[1].id, 3)
    self.assertEqual(len(tracks), 3)
    self.assertEqual(tracks[0].person.id, 2)
    self.assertEqual(tracks[0].last_timestamp, 900000)
    self.assertEqual(tracks[1].person.id, 3)
    self.assertEqual(tracks[1].last_timestamp, 900000)
    self.assertEqual(tracks[2].person.id, 1)
    self.assertEqual(tracks[2].last_timestamp, 100000)

    # Timestamp: 1200000. First person spawns a new track (id = 4), even though
    # it has the same keypoints as track 1. This is because the age exceeds
    # 1000 msec. The second person links with id 2. The third person spawns a
    # new track (id = 5).
    persons = [  # Becomes id = 4.
        Person([
            KeyPoint(BodyPart(0), Point(0.2, 0.2), 1.0),
            KeyPoint(BodyPart(1), Point(0.4, 0.4), 0.8),
            KeyPoint(BodyPart(2), Point(0.6, 0.6), 0.9),
            KeyPoint(BodyPart(3), Point(0.8, 0.8), 0.8)
        ], Rectangle(Point(0, 0), Point(0, 0)), 1),
        # Links with id = 2.
        Person(
            [
                KeyPoint(BodyPart(0), Point(0.55, 0.7), 0.7),
                KeyPoint(BodyPart(1), Point(0.5, 0.6), 0.9),
                KeyPoint(BodyPart(2), Point(1.0, 1.0), 0.1),  # Low confidence.
                KeyPoint(BodyPart(3), Point(0.8, 0.1), 0.0)
            ],  # Low confidence.
            Rectangle(Point(0, 0), Point(0, 0)),
            1),
        # Becomes id = 5.
        Person(
            [
                KeyPoint(BodyPart(0), Point(0.1, 0.1), 0.1),  # Low confidence.
                KeyPoint(BodyPart(1), Point(0.2, 0.2), 0.9),
                KeyPoint(BodyPart(2), Point(0.3, 0.3), 0.7),
                KeyPoint(BodyPart(3), Point(0.4, 0.4), 0.8)
            ],
            Rectangle(Point(0, 0), Point(0, 0)),
            1)
    ]

    persons = self.kpt_tracker.apply(persons, 1200000)
    tracks = self.kpt_tracker._tracks
    self.assertEqual(len(persons), 3)
    self.assertEqual(persons[0].id, 4)
    self.assertEqual(persons[1].id, 2)
    self.assertEqual(len(tracks), 4)
    self.assertEqual(tracks[0].person.id, 2)
    self.assertEqual(tracks[0].last_timestamp, 1200000)
    self.assertEqual(tracks[1].person.id, 4)
    self.assertEqual(tracks[1].last_timestamp, 1200000)
    self.assertEqual(tracks[2].person.id, 5)
    self.assertEqual(tracks[2].last_timestamp, 1200000)
    self.assertEqual(tracks[3].person.id, 3)
    self.assertEqual(tracks[3].last_timestamp, 900000)

    # Timestamp: 1300000. First person spawns a new track (id = 6). Since
    # max_tracks is 4, the oldest track (id = 3) is removed.
    persons = [  # Becomes id = 6.
        Person([
            KeyPoint(BodyPart(0), Point(0.1, 0.8), 1.0),
            KeyPoint(BodyPart(1), Point(0.2, 0.9), 0.6),
            KeyPoint(BodyPart(2), Point(0.2, 0.9), 0.5),
            KeyPoint(BodyPart(3), Point(0.8, 0.2), 0.4)
        ], Rectangle(Point(0, 0), Point(0, 0)), 1)
    ]

    persons = self.kpt_tracker.apply(persons, 1300000)
    tracks = self.kpt_tracker._tracks
    self.assertEqual(len(persons), 1)
    self.assertEqual(persons[0].id, 6)
    self.assertEqual(len(tracks), 4)
    self.assertEqual(tracks[0].person.id, 6)
    self.assertEqual(tracks[0].last_timestamp, 1300000)
    self.assertEqual(tracks[1].person.id, 2)
    self.assertEqual(tracks[1].last_timestamp, 1200000)
    self.assertEqual(tracks[2].person.id, 4)
    self.assertEqual(tracks[2].last_timestamp, 1200000)
    self.assertEqual(tracks[3].person.id, 5)
    self.assertEqual(tracks[3].last_timestamp, 1200000)


if __name__ == '__main__':
  unittest.main()
