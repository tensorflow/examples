# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Unit tests for the VideoClassifier wrapper."""

from typing import List
import unittest

import cv2
import numpy as np
from video_classifier import Category
from video_classifier import VideoClassifier
from video_classifier import VideoClassifierOptions

_MODEL_FILE = 'movinet_a0_int8.tflite'
_LABEL_FILE = 'kinetics600_label_map.txt'
_GROUND_TRUTH_LABEL = 'carving ice'
_GROUND_TRUTH_MIN_SCORE = 0.5
_VIDEO_FILE = 'test_data/carving_ice.mp4'
_ALLOW_LIST = ['carving ice', 'sawing wood']
_DENY_LIST = ['chiseling stone']
_SCORE_THRESHOLD = 0.2
_MAX_RESULTS = 3
_ACCEPTABLE_ERROR_RANGE = 0.01


class VideoClassifierTest(unittest.TestCase):

  def setUp(self):
    """Initialize the shared variables."""
    super().setUp()

    # Load frames from the test video.
    cap = cv2.VideoCapture(_VIDEO_FILE)
    frames = []
    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
      _, frame = cap.read()
      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      frames.append(frame)
    self._frames = frames

  def _run_classification_with_frames(self, classifier: VideoClassifier,
                                      frames: List[np.ndarray]) -> [Category]:
    """Run video classification with all the given frames and return the final result."""
    categories = None
    for frame in frames:
      categories = classifier.classify(frame)

    return categories

  def test_default_option(self):
    """Check if the default option works correctly."""
    classifier = VideoClassifier(_MODEL_FILE, _LABEL_FILE)

    # Run classification with frames from the test video.
    categories = self._run_classification_with_frames(classifier, self._frames)

    # Check if TOP-1 result match.
    top_1_category = categories[0]
    self.assertEqual(
        _GROUND_TRUTH_LABEL, top_1_category.label,
        'Label {0} does not match with ground truth {1}'.format(
            top_1_category.label, _GROUND_TRUTH_LABEL))
    self.assertLessEqual(
        _GROUND_TRUTH_MIN_SCORE, top_1_category.score,
        'Classification score {0} is smaller than threshold {1}'.format(
            top_1_category.score, _GROUND_TRUTH_MIN_SCORE))

  def test_allow_list(self):
    """Test the label_allow_list option."""
    option = VideoClassifierOptions(label_allow_list=_ALLOW_LIST)
    classifier = VideoClassifier(_MODEL_FILE, _LABEL_FILE, option)

    # Run classification with frames from the test video.
    categories = self._run_classification_with_frames(classifier, self._frames)

    for category in categories:
      label = category.label
      self.assertIn(
          label, _ALLOW_LIST,
          'Label "{0}" found but not in label allow list'.format(label))

  def test_deny_list(self):
    """Test the label_deny_list option."""
    option = VideoClassifierOptions(label_deny_list=_DENY_LIST)
    classifier = VideoClassifier(_MODEL_FILE, _LABEL_FILE, options=option)

    # Run classification with frames from the test video.
    categories = self._run_classification_with_frames(classifier, self._frames)

    for category in categories:
      label = category.label
      self.assertNotIn(label, _DENY_LIST,
                       'Label "{0}" found but in deny list.'.format(label))

  def test_score_threshold_option(self):
    """Test the score_threshold option."""
    option = VideoClassifierOptions(score_threshold=_SCORE_THRESHOLD)
    classifier = VideoClassifier(_MODEL_FILE, _LABEL_FILE, options=option)

    # Run classification with frames from the test video.
    categories = self._run_classification_with_frames(classifier, self._frames)

    for category in categories:
      score = category.score
      self.assertGreaterEqual(
          score, _SCORE_THRESHOLD,
          'Classification with score lower than threshold found. {0}'.format(
              category))

  def test_max_results_option(self):
    """Test the max_results option."""
    option = VideoClassifierOptions(max_results=_MAX_RESULTS)
    classifier = VideoClassifier(_MODEL_FILE, _LABEL_FILE, options=option)

    # Run classification with frames from the test video.
    categories = self._run_classification_with_frames(classifier, self._frames)

    self.assertLessEqual(
        len(categories), _MAX_RESULTS, 'Too many results returned.')


if __name__ == '__main__':
  unittest.main()
