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
"""Unit tests for the ImageClassifier wrapper."""

import csv
import unittest

import cv2
from image_classifier import Category
from image_classifier import ImageClassifier
from image_classifier import ImageClassifierOptions

_MODEL_FILE = 'efficientnet_lite0.tflite'
_GROUND_TRUTH_FILE = 'test_data/ground_truth.csv'
_IMAGE_FILE = 'test_data/fox.jpeg'
_ALLOW_LIST = ['red fox', 'kit fox']
_DENY_LIST = ['grey fox']
_SCORE_THRESHOLD = 0.2
_MAX_RESULTS = 3
_ACCEPTABLE_ERROR_RANGE = 0.01


class ImageClassifierTest(unittest.TestCase):

  def setUp(self):
    """Initialize the shared variables."""
    super().setUp()
    self._load_ground_truth()
    self.image = cv2.imread(_IMAGE_FILE)
    self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

  def test_default_option(self):
    """Check if the default option works correctly."""
    classifier = ImageClassifier(_MODEL_FILE)
    categories = classifier.classify(self.image)

    # Check if all ground truth classification is found.
    for gt_classification in self._ground_truth_classifications:
      is_gt_found = False
      for real_classification in categories:
        is_label_match = real_classification.label == gt_classification.label
        is_score_match = abs(real_classification.score -
                             gt_classification.score) < _ACCEPTABLE_ERROR_RANGE

        # If a matching classification is found, stop the loop.
        if is_label_match and is_score_match:
          is_gt_found = True
          break

      # If no matching classification found, fail the test.
      self.assertTrue(is_gt_found, '{0} not found.'.format(gt_classification))

  def test_allow_list(self):
    """Test the label_allow_list option."""
    option = ImageClassifierOptions(label_allow_list=_ALLOW_LIST)
    classifier = ImageClassifier(_MODEL_FILE, options=option)
    categories = classifier.classify(self.image)

    for category in categories:
      label = category.label
      self.assertIn(
          label, _ALLOW_LIST,
          'Label "{0}" found but not in label allow list'.format(label))

  def test_deny_list(self):
    """Test the label_deny_list option."""
    option = ImageClassifierOptions(label_deny_list=_DENY_LIST)
    classifier = ImageClassifier(_MODEL_FILE, options=option)
    categories = classifier.classify(self.image)

    for category in categories:
      label = category.label
      self.assertNotIn(label, _DENY_LIST,
                       'Label "{0}" found but in deny list.'.format(label))

  def test_score_threshold_option(self):
    """Test the score_threshold option."""
    option = ImageClassifierOptions(score_threshold=_SCORE_THRESHOLD)
    classifier = ImageClassifier(_MODEL_FILE, options=option)
    categories = classifier.classify(self.image)

    for category in categories:
      score = category.score
      self.assertGreaterEqual(
          score, _SCORE_THRESHOLD,
          'Classification with score lower than threshold found. {0}'.format(
              category))

  def test_max_results_option(self):
    """Test the max_results option."""
    option = ImageClassifierOptions(max_results=_MAX_RESULTS)
    classifier = ImageClassifier(_MODEL_FILE, options=option)
    categories = classifier.classify(self.image)

    self.assertLessEqual(
        len(categories), _MAX_RESULTS, 'Too many results returned.')

  def _load_ground_truth(self):
    """Load ground truth classification result from a CSV file."""
    self._ground_truth_classifications = []
    with open(_GROUND_TRUTH_FILE) as f:
      reader = csv.DictReader(f)
      for row in reader:
        category = Category(label=row['label'], score=float(row['score']))

        self._ground_truth_classifications.append(category)

# pylint: disable=g-unreachable-test-method

  def _create_ground_truth_csv(self, output_file=_GROUND_TRUTH_FILE):
    """A util function to regenerate the ground truth result.

    This function is not used in the test but it exists to make adding more
    images and ground truth data to the test easier in the future.

    Args:
      output_file: Filename to write the ground truth CSV.
    """
    classifier = ImageClassifier(_MODEL_FILE)
    categories = classifier.classify(self.image)
    with open(output_file, 'w') as f:
      header = ['label', 'score']
      writer = csv.DictWriter(f, fieldnames=header)
      writer.writeheader()
      for category in categories:
        writer.writerow({
            'label': category.label,
            'score': category.score,
        })


# pylint: enable=g-unreachable-test-method

if __name__ == '__main__':
  unittest.main()
