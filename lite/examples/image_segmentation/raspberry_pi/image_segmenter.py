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
"""Wrapper class for TensorFlow Lite image segmentation models."""

import dataclasses
import enum
import json
import platform
from typing import List, Tuple

import cv2
import numpy as np
from tflite_support import metadata

# pylint: disable=g-import-not-at-top
try:
  # Import TFLite interpreter from tflite_runtime package if it's available.
  from tflite_runtime.interpreter import Interpreter
  from tflite_runtime.interpreter import load_delegate
except ImportError:
  # If not, fallback to use the TFLite interpreter from the full TF package.
  import tensorflow as tf

  Interpreter = tf.lite.Interpreter
  load_delegate = tf.lite.experimental.load_delegate
# pylint: enable=g-import-not-at-top


class OutputType(enum.Enum):
  """Output mask type.

  This allows specifying the type of post-processing to perform on the raw
  model results.
  """

  CATEGORY_MASK = 1
  """Gives a single output mask where each pixel represents the class which the pixel in the original image was

    predicted to belong to.
  """

  CONFIDENCE_MASK = 2
  """Gives a list of output masks where, for each mask, each pixel represents the prediction confidence,

    usually in the [0, 1] range.
  """


@dataclasses.dataclass
class ImageSegmenterOptions(object):
  """A config to initialize an image segmenter."""

  enable_edgetpu: bool = False
  """Enable the model to run on EdgeTPU."""

  num_threads: int = 1
  """The number of CPU threads to be used."""

  output_type: OutputType = OutputType.CATEGORY_MASK
  """Format of the model output's segmentation mask."""


@dataclasses.dataclass
class ColoredLabel(object):
  label: str
  """The label name."""

  color: Tuple[int, int, int]
  """The RGB representation of the label's color."""


@dataclasses.dataclass
class Segmentation(object):
  colored_labels: List[ColoredLabel]
  """The map between RGB color and label name."""

  masks: np.ndarray
  """The pixel mask representing the segmentation result."""

  output_type: OutputType
  """The format of the model output."""


def edgetpu_lib_name():
  """Returns the library name of EdgeTPU in the current platform."""
  return {
      'Darwin': 'libedgetpu.1.dylib',
      'Linux': 'libedgetpu.so.1',
      'Windows': 'edgetpu.dll',
  }.get(platform.system(), None)


# A list of distinctive for visualization
# https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/vision/image_segmenter.cc#L57
_COLOR_LIST = [
    (0, 0, 0),
    (128, 0, 0),
    (0, 128, 0),
    (128, 128, 0),
    (0, 0, 128),
    (128, 0, 128),
    (0, 128, 128),
    (128, 128, 128),
    (64, 0, 0),
    (192, 0, 0),
    (64, 128, 0),
    (192, 128, 0),
    (64, 0, 128),
    (192, 0, 128),
    (64, 128, 128),
    (192, 128, 128),
    (0, 64, 0),
    (128, 64, 0),
    (0, 192, 0),
    (128, 192, 0),
    (0, 64, 128),
    (128, 64, 128),
    (0, 192, 128),
    (128, 192, 128),
    (64, 64, 0),
    (192, 64, 0),
    (64, 192, 0),
    (192, 192, 0),
    (64, 64, 128),
    (192, 64, 128),
    (64, 192, 128),
    (192, 192, 128),
    (0, 0, 64),
    (128, 0, 64),
    (0, 128, 64),
    (128, 128, 64),
    (0, 0, 192),
    (128, 0, 192),
    (0, 128, 192),
    (128, 128, 192),
    (64, 0, 64),
    (192, 0, 64),
    (64, 128, 64),
    (192, 128, 64),
    (64, 0, 192),
    (192, 0, 192),
    (64, 128, 192),
    (192, 128, 192),
    (0, 64, 64),
    (128, 64, 64),
    (0, 192, 64),
    (128, 192, 64),
    (0, 64, 192),
    (128, 64, 192),
    (0, 192, 192),
    (128, 192, 192),
    (64, 64, 64),
    (192, 64, 64),
    (64, 192, 64),
    (192, 192, 64),
    (64, 64, 192),
    (192, 64, 192),
    (64, 192, 192),
    (192, 192, 192),
    (32, 0, 0),
    (160, 0, 0),
    (32, 128, 0),
    (160, 128, 0),
    (32, 0, 128),
    (160, 0, 128),
    (32, 128, 128),
    (160, 128, 128),
    (96, 0, 0),
    (224, 0, 0),
    (96, 128, 0),
    (224, 128, 0),
    (96, 0, 128),
    (224, 0, 128),
    (96, 128, 128),
    (224, 128, 128),
    (32, 64, 0),
    (160, 64, 0),
    (32, 192, 0),
    (160, 192, 0),
    (32, 64, 128),
    (160, 64, 128),
    (32, 192, 128),
    (160, 192, 128),
    (96, 64, 0),
    (224, 64, 0),
    (96, 192, 0),
    (224, 192, 0),
    (96, 64, 128),
    (224, 64, 128),
    (96, 192, 128),
    (224, 192, 128),
    (32, 0, 64),
    (160, 0, 64),
    (32, 128, 64),
    (160, 128, 64),
    (32, 0, 192),
    (160, 0, 192),
    (32, 128, 192),
    (160, 128, 192),
    (96, 0, 64),
    (224, 0, 64),
    (96, 128, 64),
    (224, 128, 64),
    (96, 0, 192),
    (224, 0, 192),
    (96, 128, 192),
    (224, 128, 192),
    (32, 64, 64),
    (160, 64, 64),
    (32, 192, 64),
    (160, 192, 64),
    (32, 64, 192),
    (160, 64, 192),
    (32, 192, 192),
    (160, 192, 192),
    (96, 64, 64),
    (224, 64, 64),
    (96, 192, 64),
    (224, 192, 64),
    (96, 64, 192),
    (224, 64, 192),
    (96, 192, 192),
    (224, 192, 192),
    (0, 32, 0),
    (128, 32, 0),
    (0, 160, 0),
    (128, 160, 0),
    (0, 32, 128),
    (128, 32, 128),
    (0, 160, 128),
    (128, 160, 128),
    (64, 32, 0),
    (192, 32, 0),
    (64, 160, 0),
    (192, 160, 0),
    (64, 32, 128),
    (192, 32, 128),
    (64, 160, 128),
    (192, 160, 128),
    (0, 96, 0),
    (128, 96, 0),
    (0, 224, 0),
    (128, 224, 0),
    (0, 96, 128),
    (128, 96, 128),
    (0, 224, 128),
    (128, 224, 128),
    (64, 96, 0),
    (192, 96, 0),
    (64, 224, 0),
    (192, 224, 0),
    (64, 96, 128),
    (192, 96, 128),
    (64, 224, 128),
    (192, 224, 128),
    (0, 32, 64),
    (128, 32, 64),
    (0, 160, 64),
    (128, 160, 64),
    (0, 32, 192),
    (128, 32, 192),
    (0, 160, 192),
    (128, 160, 192),
    (64, 32, 64),
    (192, 32, 64),
    (64, 160, 64),
    (192, 160, 64),
    (64, 32, 192),
    (192, 32, 192),
    (64, 160, 192),
    (192, 160, 192),
    (0, 96, 64),
    (128, 96, 64),
    (0, 224, 64),
    (128, 224, 64),
    (0, 96, 192),
    (128, 96, 192),
    (0, 224, 192),
    (128, 224, 192),
    (64, 96, 64),
    (192, 96, 64),
    (64, 224, 64),
    (192, 224, 64),
    (64, 96, 192),
    (192, 96, 192),
    (64, 224, 192),
    (192, 224, 192),
    (32, 32, 0),
    (160, 32, 0),
    (32, 160, 0),
    (160, 160, 0),
    (32, 32, 128),
    (160, 32, 128),
    (32, 160, 128),
    (160, 160, 128),
    (96, 32, 0),
    (224, 32, 0),
    (96, 160, 0),
    (224, 160, 0),
    (96, 32, 128),
    (224, 32, 128),
    (96, 160, 128),
    (224, 160, 128),
    (32, 96, 0),
    (160, 96, 0),
    (32, 224, 0),
    (160, 224, 0),
    (32, 96, 128),
    (160, 96, 128),
    (32, 224, 128),
    (160, 224, 128),
    (96, 96, 0),
    (224, 96, 0),
    (96, 224, 0),
    (224, 224, 0),
    (96, 96, 128),
    (224, 96, 128),
    (96, 224, 128),
    (224, 224, 128),
    (32, 32, 64),
    (160, 32, 64),
    (32, 160, 64),
    (160, 160, 64),
    (32, 32, 192),
    (160, 32, 192),
    (32, 160, 192),
    (160, 160, 192),
    (96, 32, 64),
    (224, 32, 64),
    (96, 160, 64),
    (224, 160, 64),
    (96, 32, 192),
    (224, 32, 192),
    (96, 160, 192),
    (224, 160, 192),
    (32, 96, 64),
    (160, 96, 64),
    (32, 224, 64),
    (160, 224, 64),
    (32, 96, 192),
    (160, 96, 192),
    (32, 224, 192),
    (160, 224, 192),
    (96, 96, 64),
    (224, 96, 64),
    (96, 224, 64),
    (224, 224, 64),
    (96, 96, 192),
    (224, 96, 192),
    (96, 224, 192),
    (224, 224, 192),
]


def _label_to_color_image(label: np.ndarray) -> np.ndarray:
  """Adds color defined by the dataset colormap to the label.

  Args:
      label: A 2D array with integer type, storing the segmentation label.

  Returns:
      A 2D array with floating type. The element of the array is the color
      indexed by the corresponding element in the input label to color map.

  Raises:
      ValueError: If label is not of rank 2 or its value is larger than
      color map maximum entry.
  """
  if label.ndim != 2:
    raise ValueError('Expect 2-D input label')

  if np.max(label) >= len(_COLOR_LIST):
    raise ValueError('Label value too large.')

  return _COLOR_LIST[label]


class ImageSegmenter(object):
  """A wrapper class for a TFLite image segmentation model."""

  def __init__(
      self,
      model_path: str,
      options: ImageSegmenterOptions = ImageSegmenterOptions()
  ) -> None:
    """Initialize a image segmentation model.

    Args:
        model_path: Name of the TFLite image segmentation model.
        options: The config to initialize an image segmenter. (Optional)

    Raises:
        ValueError: If the TFLite model is invalid.
        OSError: If the current OS isn't supported by EdgeTPU.
    """
    # Load metadata from model.
    displayer = metadata.MetadataDisplayer.with_model_file(model_path)

    # Save model metadata for preprocessing later.
    model_metadata = json.loads(displayer.get_metadata_json())
    process_units = model_metadata['subgraph_metadata'][0][
        'input_tensor_metadata'][0]['process_units']

    mean = 127.5
    std = 127.5
    for option in process_units:
      if option['options_type'] == 'NormalizationOptions':
        mean = option['options']['mean'][0]
        std = option['options']['std'][0]
    self._mean = mean
    self._std = std

    # Load label list from metadata.
    file_name = displayer.get_packed_associated_file_list()[0]
    label_map_file = displayer.get_associated_file_buffer(file_name).decode()
    label_list = list(filter(len, label_map_file.splitlines()))
    self._label_list = label_list

    # Initialize TFLite model.
    if options.enable_edgetpu:
      if edgetpu_lib_name() is None:
        raise OSError("The current OS isn't supported by Coral EdgeTPU.")
      interpreter = Interpreter(
          model_path=model_path,
          experimental_delegates=[load_delegate(edgetpu_lib_name())],
          num_threads=options.num_threads)
    else:
      interpreter = Interpreter(
          model_path=model_path, num_threads=options.num_threads)
    interpreter.allocate_tensors()

    self._options = options
    self._input_index = interpreter.get_input_details()[0]['index']
    self._output_index = interpreter.get_output_details()[0]['index']

    self._input_height = interpreter.get_input_details()[0]['shape'][1]
    self._input_width = interpreter.get_input_details()[0]['shape'][2]

    self._is_quantized_input = interpreter.get_input_details(
    )[0]['dtype'] == np.uint8

    self._interpreter = interpreter

  def _preprocess(self, input_image: np.ndarray) -> np.ndarray:
    """Preprocess the image as required by the TFLite model."""
    input_tensor = cv2.resize(input_image,
                              (self._input_width, self._input_height))
    # Normalize the input if it's a float model (aka. not quantized)
    if not self._is_quantized_input:
      input_tensor = (np.float32(input_tensor) - self._mean) / self._std

    return input_tensor

  def _set_input_tensor(self, image: np.ndarray) -> None:
    """Sets the input tensor."""
    tensor_index = self._input_index
    input_tensor = self._interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image

  def segment(self, input_image: np.ndarray) -> Segmentation:
    """Run segmentation on an input image.

    Args:
        input_image: A [height, width, 3] RGB image.

    Returns:
        Segmentation output.
    """
    input_tensor = self._preprocess(input_image)
    self._set_input_tensor(input_tensor)
    self._interpreter.invoke()

    output_tensor = self._interpreter.get_tensor(self._output_index)

    return self._postprocess(output_tensor)

  def _postprocess(self, output_tensor: np.ndarray) -> Segmentation:
    """Post-process the output tensor into segmentation output.

    Args:
        output_tensor: Output tensor of TFLite model.

    Returns:
        Segmentation output.
    """
    output_tensor = np.squeeze(output_tensor)

    if len(output_tensor.shape) == 2:
      # If the model outputs category mask, force output type to be
      # CONFIDENCE_MASK.
      output_type = OutputType.CATEGORY_MASK
    else:
      # If the model outputs confidence mask, use the output type specified in
      # the initialization option.
      output_type = self._options.output_type
      if output_type == OutputType.CATEGORY_MASK:
        output_tensor = np.argmax(output_tensor, axis=2)

    # Get label_name and color_map from label_index
    colored_labels = []
    for idx in range(len(self._label_list)):
      colored_labels.append(
          ColoredLabel(
              label=self._label_list[idx],
              color=_COLOR_LIST[idx % len(_COLOR_LIST)]))

    return Segmentation(
        colored_labels=colored_labels,
        masks=output_tensor,
        output_type=output_type)
