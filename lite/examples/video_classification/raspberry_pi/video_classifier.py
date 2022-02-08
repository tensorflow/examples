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
"""A wrapper for TensorFlow Lite video classification models."""

from typing import List, NamedTuple

import cv2
import numpy as np

# pylint: disable=g-import-not-at-top
try:
  # Import TFLite interpreter from tflite_runtime package if it's available.
  from tflite_runtime.interpreter import Interpreter
except ImportError:
  # If not, fallback to use the TFLite interpreter from the full TF package.
  import tensorflow as tf
  Interpreter = tf.lite.Interpreter
# pylint: enable=g-import-not-at-top


class VideoClassifierOptions(NamedTuple):
  """A config to initialize an video classifier."""

  label_allow_list: List[str] = None
  """The optional allow list of labels."""

  label_deny_list: List[str] = None
  """The optional deny list of labels."""

  max_results: int = 5
  """The maximum number of top-scored classification results to return."""

  num_threads: int = 1
  """The number of CPU threads to be used."""

  score_threshold: float = 0.0
  """The score threshold of classification results to return."""


class Category(NamedTuple):
  """A result of a video classification."""
  label: str
  score: float


class VideoClassifier(object):
  """A wrapper class for a TFLite video classification model."""

  _MODEL_INPUT_SIGNATURE_NAME = 'image'
  _MODEL_OUTPUT_SIGNATURE_NAME = 'logits'
  _MODEL_INPUT_MEAN = 0
  _MODEL_INPUT_STD = 255

  def __init__(
      self,
      model_path: str,
      label_file: str,
      options: VideoClassifierOptions = VideoClassifierOptions()
  ) -> None:
    """Initialize a video classification model.

    Args:
        model_path: Path of the TFLite video classification model.
        label_file: Path of the video classification label list.
        options: The config to initialize an video classifier. (Optional)

    Raises:
        ValueError: If the TFLite model is invalid.
    """

    interpreter = Interpreter(
        model_path=model_path, num_threads=options.num_threads)
    signature = interpreter.get_signature_runner()

    # Load the label list.
    with open(label_file, 'r') as f:
      lines = f.readlines()
      label_list = [line.replace('\n', '') for line in lines]
      self._label_list = label_list

    # Remove the batch dimension to get the real input shape.
    input_shape = signature.get_input_details()[
        self._MODEL_INPUT_SIGNATURE_NAME]['shape']
    input_shape = np.delete(input_shape, np.where(input_shape == 1))
    self._input_height = input_shape[0]
    self._input_width = input_shape[1]

    # Store the signature runner and model options for later use.
    self._signature = signature
    self._options = options

    # Set the initial state for the model.
    self._internal_states = {}
    self.clear()

  def clear(self):
    """Clear the internal state of the model to start classifying a new scene."""
    # Create the initial (zero) states
    init_states = {
        name: np.zeros(signature['shape'], dtype=signature['dtype'])
        for name, signature in self._signature.get_input_details().items()
    }

    # Remove the holder for the input image as it'll be fed by the caller.
    init_states.pop(self._MODEL_INPUT_SIGNATURE_NAME)

    # Store the model's internal state.
    self._internal_states = init_states

  def _preprocess(self, image: np.ndarray) -> np.ndarray:
    """Preprocess the image as required by the TFLite model."""
    input_tensor = cv2.resize(image, (self._input_width, self._input_height))
    input_tensor = input_tensor[np.newaxis, np.newaxis]
    input_tensor = np.float32(input_tensor -
                              self._MODEL_INPUT_MEAN) / self._MODEL_INPUT_STD

    return input_tensor

  def classify(self, frame: np.ndarray) -> List[Category]:
    """Classify an input frame.

    Frames from the target video should be fed to the model in sequence.

    Args:
        frame: A [height, width, 3] RGB image representing a frame in a video.

    Returns:
        A list of prediction result. Sorted by probability descending.
    """
    # Preprocess the input frame.
    frame = self._preprocess(frame)

    # Feed the input frame and the model internal states to the TFLite model.
    outputs = self._signature(**self._internal_states, image=frame)

    # Take the model output and store the internal states for subsequence
    # frames.
    logits = outputs.pop(self._MODEL_OUTPUT_SIGNATURE_NAME)
    self._internal_states = outputs

    return self._postprocess(logits)

  def _postprocess(self, logits: np.ndarray) -> List[Category]:
    """Post-process the logits into a list of Category objects.

    Args:
        logits: Raw logits output of the TFLite model.

    Returns:
        A list of classification results.
    """
    # Convert from logits to probabilities using softmax function.
    exp_logits = np.exp(np.squeeze(logits, axis=0))
    probabilities = exp_logits / np.sum(exp_logits)

    # Sort the labels so that the more likely categories come first.
    prob_descending = sorted(
        range(len(probabilities)), key=lambda k: probabilities[k], reverse=True)
    categories = [
        Category(label=self._label_list[idx], score=probabilities[idx])
        for idx in prob_descending
    ]

    # Filter out categories in the deny list.
    filtered_results = categories
    if self._options.label_deny_list is not None:
      filtered_results = list(
          filter(
              lambda category: category.label not in self._options.
              label_deny_list, filtered_results))

    # Keep only categories in the allow list.
    if self._options.label_allow_list is not None:
      filtered_results = list(
          filter(
              lambda category: category.label in self._options.label_allow_list,
              filtered_results))

    # Filter out categories with score lower than the score threshold.
    if self._options.score_threshold is not None:
      filtered_results = list(
          filter(
              lambda category: category.score >= self._options.score_threshold,
              filtered_results))

    # Only return maximum of max_results categories.
    if self._options.max_results > 0:
      result_count = min(len(filtered_results), self._options.max_results)
      filtered_results = filtered_results[:result_count]

    return filtered_results
