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
"""Unit tests for the AudioClassifier wrapper."""

import csv
import unittest

from audio_classifier import AudioClassifier
from audio_classifier import AudioClassifierOptions
from audio_classifier import Category
import numpy as np
from scipy.io import wavfile

_MODEL_FILE = 'yamnet.tflite'
_GROUND_TRUTH_FILE = 'test_data/ground_truth.csv'
_AUDIO_FILE = 'test_data/meow_16k.wav'
_ACCEPTABLE_ERROR_RANGE = 0.01


class AudioClassifierTest(unittest.TestCase):

  def setUp(self):
    """Initialize the shared variables."""
    super().setUp()
    self._load_ground_truth()

    # Load the TFLite model to get the audio format required by the model.
    classifier = AudioClassifier(_MODEL_FILE)
    tensor = classifier.create_input_tensor_audio()
    input_size = len(tensor.buffer)
    input_sample_rate = tensor.format.sample_rate
    channels = tensor.format.channels

    # Load the input audio file. Use only the beginning of the file that fits
    # the model input size.
    original_sample_rate, wav_data = wavfile.read(_AUDIO_FILE, True)

    # Ensure that the WAV file's sampling rate matches with the model
    # requirement.
    self.assertEqual(
        original_sample_rate, input_sample_rate,
        'The test audio\'s sample rate does not match with the model\'s requirement.'
    )

    # Normalize to [-1, 1] and cast to float32
    wav_data = (wav_data / np.iinfo(wav_data.dtype).max).astype(np.float32)

    # Use only the beginning of the file that fits the model input size.
    wav_data = np.reshape(wav_data[:input_size], [input_size, channels])
    tensor.load_from_array(wav_data)
    self._input_tensor = tensor

  def test_default_option(self):
    """Check if the default option works correctly."""
    classifier = AudioClassifier(_MODEL_FILE)
    categories = classifier.classify(self._input_tensor)

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
    allow_list = ['Cat']
    option = AudioClassifierOptions(label_allow_list=allow_list)
    classifier = AudioClassifier(_MODEL_FILE, options=option)
    categories = classifier.classify(self._input_tensor)

    for category in categories:
      label = category.label
      self.assertIn(
          label, allow_list,
          'Label "{0}" found but not in label allow list'.format(label))

  def test_deny_list(self):
    """Test the label_deny_list option."""
    deny_list = ['Animal']
    option = AudioClassifierOptions(label_deny_list=deny_list)
    classifier = AudioClassifier(_MODEL_FILE, options=option)
    categories = classifier.classify(self._input_tensor)

    for category in categories:
      label = category.label
      self.assertNotIn(label, deny_list,
                       'Label "{0}" found but in deny list.'.format(label))

  def test_score_threshold_option(self):
    """Test the score_threshold option."""
    score_threshold = 0.5
    option = AudioClassifierOptions(score_threshold=score_threshold)
    classifier = AudioClassifier(_MODEL_FILE, options=option)
    categories = classifier.classify(self._input_tensor)

    for category in categories:
      score = category.score
      self.assertGreaterEqual(
          score, score_threshold,
          'Classification with score lower than threshold found. {0}'.format(
              category))

  def test_max_results_option(self):
    """Test the max_results option."""
    max_results = 3
    option = AudioClassifierOptions(max_results=max_results)
    classifier = AudioClassifier(_MODEL_FILE, options=option)
    categories = classifier.classify(self._input_tensor)

    self.assertLessEqual(
        len(categories), max_results, 'Too many results returned.')

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
    audio and ground truth data to the test easier in the future.

    Args:
      output_file: Filename to write the ground truth CSV.
    """
    classifier = AudioClassifier(_MODEL_FILE)
    categories = classifier.classify(self._input_tensor)
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
