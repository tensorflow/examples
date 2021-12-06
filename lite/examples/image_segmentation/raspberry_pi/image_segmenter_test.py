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
"""Unit test of image segmentation using ImageSegmenter wrapper."""

import unittest

import cv2
from image_segmenter import ImageSegmenter
from image_segmenter import ImageSegmenterOptions
from image_segmenter import OutputType
import numpy as np
import utils

_MODEL_FILE = 'deeplabv3.tflite'
_IMAGE_FILE = 'test_data/input_image.jpeg'
_GROUND_TRUTH_IMAGE_FILE = 'test_data/ground_truth_segmentation.png'
_GROUND_TRUTH_LABEL_FILE = 'test_data/ground_truth_label.txt'
_MATCH_PIXELS_THRESHOLD = 0.01


class ImageSegmenterTest(unittest.TestCase):

  def _load_ground_truth(self):
    """Load ground truth segmentation result from the image and CSV file."""
    self._ground_truth_segmentation = cv2.imread(_GROUND_TRUTH_IMAGE_FILE)
    self._ground_truth_labels = []
    with open(_GROUND_TRUTH_LABEL_FILE) as f:
      self._ground_truth_labels = f.read().splitlines()

  def setUp(self):
    """Initialize the shared variables."""
    super().setUp()
    self._load_ground_truth()
    image = cv2.imread(_IMAGE_FILE)
    self.image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Run segmentation with the TFLite model in CATEGORY_MASK mode.
    segmenter = ImageSegmenter(_MODEL_FILE)
    result = segmenter.segment(self.image)
    self._seg_map_img, self._found_labels = utils.segmentation_map_to_image(
        result)
    self._category_mask = result.masks

  def test_segmentation_category_mask(self):
    """Check if category mask match with ground truth."""
    result_pixels = self._seg_map_img.flatten()
    ground_truth_pixels = self._ground_truth_segmentation.flatten()

    self.assertEqual(
        len(result_pixels), len(ground_truth_pixels),
        'Segmentation mask must have the same size as ground truth.')

    inconsistent_pixels = [
        1 for idx in range(len(result_pixels))
        if result_pixels[idx] != ground_truth_pixels[idx]
    ]

    self.assertLessEqual(
        len(inconsistent_pixels) / len(result_pixels), _MATCH_PIXELS_THRESHOLD,
        'Size of the segmentation mask must be the same as ground truth.')

  def test_segmentation_confidence_mask(self):
    """Check if confidence mask matches with category mask."""
    # Run segmentation with the TFLite model in CONFIDENCE_MASK mode.
    options = ImageSegmenterOptions(output_type=OutputType.CONFIDENCE_MASK)
    segmenter = ImageSegmenter(_MODEL_FILE, options)
    result = segmenter.segment(self.image)

    # Check if confidence mask shape is correct.
    self.assertEqual(
        result.masks.shape[2], len(result.colored_labels),
        '3rd dimension of confidence mask must match with number of categories.'
    )

    calculated_category_mask = np.argmax(result.masks, axis=2)
    self.assertListEqual(calculated_category_mask.tolist(),
                         self._category_mask.tolist())

  def test_labels(self):
    """Check if detected labels match with ground truth labels."""
    result_labels = [
        colored_label.label for colored_label in self._found_labels
    ]
    self.assertEqual(result_labels, self._ground_truth_labels)

# pylint: disable=g-unreachable-test-method

  def _create_ground_truth_data(
      self,
      output_image_file: str = _GROUND_TRUTH_IMAGE_FILE,
      output_label_file: str = _GROUND_TRUTH_LABEL_FILE) -> None:
    """A util function to generate the ground truth result.

    Args:
      output_image_file: Path to save the segmentation map of output model.
      output_label_file: Path to save the label list of output model.
    """

    # Initialize the image segmentation model
    segmenter = ImageSegmenter(_MODEL_FILE)
    result = segmenter.segment(self.image)
    seg_map_img, found_labels = utils.segmentation_map_to_image(result)
    cv2.imwrite(output_image_file, seg_map_img)

    with open(output_label_file, 'w') as f:
      f.writelines('\n'.join(
          [color_label.label for color_label in found_labels]))


# pylint: enable=g-unreachable-test-method

if __name__ == '__main__':
  unittest.main()
