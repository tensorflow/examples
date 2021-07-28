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
"""Code to run a pose estimation with a TFLite PoseNet model."""

import os
import cv2
import numpy as np

# pylint: disable=g-import-not-at-top
try:
  # Import TFLite interpreter from tflite_runtime package if it's available.
  from tflite_runtime.interpreter import Interpreter
except ImportError:
  # If not, fallback to use the TFLite interpreter from the full TF package.
  from tensorflow.lite import Interpreter
# pylint: enable=g-import-not-at-top


class Posenet(object):
  """A wrapper class for a Posenet TFLite pose estimation model."""

  def __init__(self, model_name):
    """Initialize a PoseNet pose estimation model.

    Args:
        model_name: Name of the TFLite PoseNet model.
    """
    # Append TFLITE extension to model_name if there's no extension
    _, ext = os.path.splitext(model_name)
    if not ext:
      model_name += '.tflite'

    # Initialize model
    interpreter = Interpreter(model_path=model_name, num_threads=4)

    self._input_index = interpreter.get_input_details()[0]['index']
    self._output_heatmap_index = interpreter.get_output_details()[0]['index']
    self._output_offset_index = interpreter.get_output_details()[1]['index']

    self._input_height = interpreter.get_input_details()[0]['shape'][1]
    self._input_width = interpreter.get_input_details()[0]['shape'][2]

    self._interpreter = interpreter

  def detect(self, input_image):
    """Run detection on an input image.

    Args:
        input_image: A [height, width, 3] RGB image. Note that height and width
          can be anything since the image will be immediately resized according
          to the needs of the model within this function.

    Returns:
        An array of shape [17, 3] representing the keypoint coordinates and
        scores.
    """

    input_image = cv2.resize(input_image,
                             (self._input_width, self._input_height))

    input_tensor = np.expand_dims(input_image, axis=0)

    # check the type of the input tensor
    is_float_model = self._interpreter.get_input_details(
    )[0]['dtype'] == np.float32

    if is_float_model:
      input_tensor = (np.float32(input_tensor) - 127.5) / 127.5

    # Process template image
    # Sets the value of the input tensor
    self._interpreter.set_tensor(self._input_index, input_tensor)
    # Runs the computation
    self._interpreter.invoke()

    # Extract output data from the interpreter
    raw_heatmap = self._interpreter.get_tensor(self._output_heatmap_index)
    raw_offset = self._interpreter.get_tensor(self._output_offset_index)

    # Getting rid of the extra dimension
    raw_heatmap = np.squeeze(raw_heatmap)
    raw_offset = np.squeeze(raw_offset)

    keypoints_with_scores = self._process_output(raw_heatmap, raw_offset)

    return keypoints_with_scores

  def _sigmoid(self, x):
    return 1 / (1 + np.exp(-x))

  def _process_output(self, heatmap_data, offset_data):
    """Post-process the output of Posenet TFLite model.

    Args:
      heatmap_data: heatmaps output from Posenet. [height_resolution,
        width_resolution, 17]
      offset_data: offset vectors (XY) output from Posenet. [height_resolution,
        width_resolution, 34]

    Returns:
      An array of shape [17, 3] representing the keypoint absolute coordinates
      and scores.
    """
    joint_num = heatmap_data.shape[-1]
    keypoints_with_scores = np.zeros((joint_num, 3), np.float32)
    scores = self._sigmoid(heatmap_data)

    for idx in range(joint_num):
      joint_heatmap = heatmap_data[..., idx]
      x, y = np.unravel_index(
          np.argmax(scores[:, :, idx]), scores[:, :, idx].shape)
      max_val_pos = np.squeeze(
          np.argwhere(joint_heatmap == np.max(joint_heatmap)))
      remap_pos = np.array(max_val_pos / 8 * 257, dtype=np.int32)

      keypoints_with_scores[idx, 0] = (
          remap_pos[0] + offset_data[max_val_pos[0], max_val_pos[1], idx]) / 257
      keypoints_with_scores[idx, 1] = (
          remap_pos[1] +
          offset_data[max_val_pos[0], max_val_pos[1], idx + joint_num]) / 257
      keypoints_with_scores[idx, 2] = scores[x, y, idx]

    return keypoints_with_scores
