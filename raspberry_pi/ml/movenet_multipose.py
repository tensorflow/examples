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
"""Code to run a pose estimation with a TFLite Movenet_multipose model."""

import os
import time
from typing import List

import cv2
from data import BodyPart
from data import KeyPoint
from data import Person
from data import Point
from data import Rectangle
import numpy as np
from tracker import BoundingBoxTracker
from tracker import KeypointTracker
from tracker import TrackerConfig
import utils

# pylint: disable=g-import-not-at-top
try:
  # Import TFLite interpreter from tflite_runtime package if it's available.
  from tflite_runtime.interpreter import Interpreter
except ImportError:
  # If not, fallback to use the TFLite interpreter from the full TF package.
  import tensorflow as tf
  Interpreter = tf.lite.Interpreter


class MoveNetMultiPose(object):
  """A wrapper class for a MultiPose TFLite pose estimation model."""

  def __init__(self,
               model_name: str,
               tracker_type: str = 'bounding_box',
               input_size: int = 256) -> None:
    """Initialize a MultiPose pose estimation model.

    Args:
      model_name: Name of the TFLite multipose model.
      tracker_type: Type of Tracker('keypoint' or 'bounding_box')
      input_size: Size of the longer dimension of the input image.
    """
    # Append .tflite extension to model_name if there's no extension.
    _, ext = os.path.splitext(model_name)
    if not ext:
      model_name += '.tflite'

    # Store the input size parameter.
    self._input_size = input_size

    # Initialize the TFLite model.
    interpreter = Interpreter(model_path=model_name, num_threads=4)

    self._input_details = interpreter.get_input_details()
    self._output_details = interpreter.get_output_details()
    self._input_type = self._input_details[0]['dtype']

    self._input_height = interpreter.get_input_details()[0]['shape'][1]
    self._input_width = interpreter.get_input_details()[0]['shape'][2]

    self._interpreter = interpreter

    # Initialize a tracker.
    config = TrackerConfig()
    if tracker_type == 'keypoint':
      self._tracker = KeypointTracker(config)
    elif tracker_type == 'bounding_box':
      self._tracker = BoundingBoxTracker(config)
    else:
      print('ERROR: Tracker type {0} not supported. No tracker will be used.'
            .format(tracker_type))
      self._tracker = None

  def detect(self,
             input_image: np.ndarray,
             detection_threshold: float = 0.11) -> List[Person]:
    """Run detection on an input image.

    Args:
      input_image: A [height, width, 3] RGB image. Note that height and width
        can be anything since the image will be immediately resized according to
        the needs of the model within this function.
      detection_threshold: minimum confidence score for an detected pose to be
        considered.

    Returns:
      A list of Person instances detected from the input image.
    """

    is_dynamic_shape_model = self._input_details[0]['shape_signature'][2] == -1
    # Resize and pad the image to keep the aspect ratio and fit the expected
    # size.
    if is_dynamic_shape_model:
      resized_image, _ = utils.keep_aspect_ratio_resizer(
          input_image, self._input_size)
      input_tensor = np.expand_dims(resized_image, axis=0)
      self._interpreter.resize_tensor_input(
          self._input_details[0]['index'], input_tensor.shape, strict=True)
    else:
      resized_image = cv2.resize(input_image,
                                 (self._input_width, self._input_height))
      input_tensor = np.expand_dims(resized_image, axis=0)
    self._interpreter.allocate_tensors()

    # Run inference with the MoveNet MultiPose model.
    self._interpreter.set_tensor(self._input_details[0]['index'],
                                 input_tensor.astype(self._input_type))
    self._interpreter.invoke()

    # Get the model output
    model_output = self._interpreter.get_tensor(
        self._output_details[0]['index'])

    image_height, image_width, _ = input_image.shape
    return self._postprocess(model_output, image_height, image_width,
                             detection_threshold)

  def _postprocess(self, keypoints_with_scores: np.ndarray, image_height: int,
                   image_width: int,
                   detection_threshold: float) -> List[Person]:
    """Returns a list "Person" corresponding to the input image.

    Note that coordinates are expressed in (x, y) format for drawing
    utilities.

    Args:
      keypoints_with_scores: Output of the MultiPose TFLite model.
      image_height: height of the image in pixels.
      image_width: width of the image in pixels.
      detection_threshold: minimum confidence score for an entity to be
        considered.

    Returns:
      A list of Person(keypoints, bounding_box, scores), each containing:
        * the coordinates of all keypoints of the detected entity;
        * the bounding boxes of the entity.
        * the confidence core of the entity.
    """

    _, num_instances, _ = keypoints_with_scores.shape
    list_persons = []
    for idx in range(num_instances):
      # Skip a detected pose if its confidence score is below the threshold
      person_score = keypoints_with_scores[0, idx, 55]
      if person_score < detection_threshold:
        continue

      # Extract the keypoint coordinates and scores
      kpts_y = keypoints_with_scores[0, idx, range(0, 51, 3)]
      kpts_x = keypoints_with_scores[0, idx, range(1, 51, 3)]
      scores = keypoints_with_scores[0, idx, range(2, 51, 3)]

      # Create the list of keypoints
      keypoints = []
      for i in range(scores.shape[0]):
        keypoints.append(
            KeyPoint(
                BodyPart(i),
                Point(
                    int(kpts_x[i] * image_width),
                    int(kpts_y[i] * image_height)), scores[i]))

      # Calculate the bounding box
      rect = [
          keypoints_with_scores[0, idx, 51], keypoints_with_scores[0, idx, 52],
          keypoints_with_scores[0, idx, 53], keypoints_with_scores[0, idx, 54]
      ]
      bounding_box = Rectangle(
          Point(int(rect[1] * image_width), int(rect[0] * image_height)),
          Point(int(rect[3] * image_width), int(rect[2] * image_height)))

      # Create a Person instance corresponding to the detected entity.
      list_persons.append(Person(keypoints, bounding_box, person_score))
    if self._tracker:
      list_persons = self._tracker.apply(list_persons, time.time() * 1000)

    return list_persons
