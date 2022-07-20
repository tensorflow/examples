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
"""Unit test of Bounding box tracker."""

import unittest

from data import Person
from data import Point
from data import Rectangle
from tracker.bounding_box_tracker import BoundingBoxTracker
from tracker.config import KeypointTrackerConfig
from tracker.config import TrackerConfig
from tracker.tracker import Track

_MAX_TRACKS = 4
_MAX_AGE = 1000 * 1000
_MIN_SIMILARITY = 0.5


class BoundingBoxTrackerTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.tracker_config = TrackerConfig(KeypointTrackerConfig(), _MAX_TRACKS,
                                        _MAX_AGE, _MIN_SIMILARITY)
    self.bbox_tracker = BoundingBoxTracker(self.tracker_config)

  def test_iou(self):
    """Test IoU."""
    person = Person([], Rectangle(Point(0, 0), Point(1, 2 / 3)), 1)
    track = Track(
        Person([], Rectangle(Point(0, 1 / 3), Point(1, 1)), 1), 1000000)
    computed_iou = self.bbox_tracker._iou(person, track)
    self.assertAlmostEqual(computed_iou, 1 / 3, 6)

  def test_iou_full_overlap(self):
    """Test IoU full overlap."""
    person = Person([], Rectangle(Point(0, 0), Point(1, 1)), 1)
    track = Track(Person([], Rectangle(Point(0, 0), Point(1, 1)), 1), 1000000)
    computed_iou = self.bbox_tracker._iou(person, track)
    self.assertAlmostEqual(computed_iou, 1.0, 6)

  def test_iou_no_intersection(self):
    """Test IoU with no intersection."""
    person = Person([], Rectangle(Point(0, 0), Point(0.5, 0.5)), 1)
    track = Track(
        Person([], Rectangle(Point(0.5, 0.5), Point(1, 1)), 1), 1000000)
    computed_iou = self.bbox_tracker._iou(person, track)
    self.assertAlmostEqual(computed_iou, 0.0, 6)

  def test_bounding_box_tracker(self):
    """Test BoundingBoxTracker."""

    # Timestamp: 0. Person becomes the first two tracks
    persons = [
        Person([], Rectangle(Point(0, 0), Point(0.5, 0.5)), 1),
        Person(
            [],
            Rectangle(Point(0, 0), Point(1, 1)),
            1,
        )
    ]
    persons = self.bbox_tracker.apply(persons, 0)
    tracks = self.bbox_tracker._tracks
    self.assertEqual(len(persons), 2)
    self.assertEqual(persons[0].id, 1)
    self.assertEqual(persons[1].id, 2)
    self.assertEqual(len(tracks), 2)
    self.assertEqual(tracks[0].person.id, 1)
    self.assertEqual(tracks[0].last_timestamp, 0)
    self.assertEqual(tracks[1].person.id, 2)
    self.assertEqual(tracks[1].last_timestamp, 0)

    # Timestamp: 100000. First person is linked with track 1. Second person
    # spawns a new track (id = 2).
    persons = [
        Person([], Rectangle(Point(0.1, 0.1), Point(0.5, 0.5)), 1),
        Person([], Rectangle(Point(0.3, 0.2), Point(0.9, 0.9)), 1)
    ]
    persons = self.bbox_tracker.apply(persons, 100000)
    tracks = self.bbox_tracker._tracks
    self.assertEqual(len(persons), 2)
    self.assertEqual(persons[0].id, 1)
    self.assertEqual(persons[1].id, 3)
    self.assertEqual(len(tracks), 3)
    self.assertEqual(tracks[0].person.id, 1)
    self.assertEqual(tracks[0].last_timestamp, 100000)
    self.assertEqual(tracks[1].person.id, 3)
    self.assertEqual(tracks[1].last_timestamp, 100000)
    self.assertEqual(tracks[2].person.id, 2)
    self.assertEqual(tracks[2].last_timestamp, 0)

    # Timestamp: 1050000. First person is linked with track 1. Second person is
    # identical to track 2, but is not linked because track 2 is deleted due to
    # age. Instead it spawns track 4.
    persons = [
        Person([], Rectangle(Point(0.1, 0.1), Point(0.5, 0.55)), 1),
        Person([], Rectangle(Point(0, 0), Point(1, 1)), 1)
    ]
    persons = self.bbox_tracker.apply(persons, 1050000)
    tracks = self.bbox_tracker._tracks
    self.assertEqual(len(persons), 2)
    self.assertEqual(persons[0].id, 1)
    self.assertEqual(persons[1].id, 4)
    self.assertEqual(len(tracks), 3)
    self.assertEqual(tracks[0].person.id, 1)
    self.assertEqual(tracks[0].last_timestamp, 1050000)
    self.assertEqual(tracks[1].person.id, 4)
    self.assertEqual(tracks[1].last_timestamp, 1050000)
    self.assertEqual(tracks[2].person.id, 3)
    self.assertEqual(tracks[2].last_timestamp, 100000)


if __name__ == '__main__':
  unittest.main()
