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
"""A module to run audio classification with a TensorFlow Lite model."""
import json
import platform
from typing import List, NamedTuple

from audio_record import AudioRecord
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


class AudioClassifierOptions(NamedTuple):
  """A config to initialize an audio classifier."""

  enable_edgetpu: bool = False
  """Enable the model to run on EdgeTPU."""

  label_allow_list: List[str] = None
  """The optional allow list of labels."""

  label_deny_list: List[str] = None
  """The optional deny list of labels."""

  max_results: int = 5
  """The maximum number of top-scored classification results to return."""

  num_threads: int = 4
  """The number of CPU threads to be used."""

  score_threshold: float = 0.0
  """The score threshold of classification results to return."""


class Category(NamedTuple):
  """A result of a audio classification."""
  label: str
  score: float


class AudioFormat(NamedTuple):
  """Format of the incoming audio."""
  channels: int
  sample_rate: int


class TensorAudio(object):
  """A wrapper class to store the input audio."""

  def __init__(self, audio_format: AudioFormat, sample_count: int):
    self._format = audio_format
    self._sample_count = sample_count
    self.clear()

  @property
  def format(self) -> AudioFormat:
    return self._format

  def clear(self):
    """Clear the internal buffer and fill it with zeros."""
    self._buffer = np.zeros([self._sample_count, self._format.channels])

  def load_from_audio_record(self, audio_record: AudioRecord) -> None:
    """Load audio data from an AudioRecord instance.

    If the audio recorder buffer has more data than this TensorAudio can store,
    it'll take only the last bytes (most recent audio data) that fits its
    buffer.

    Args:
      audio_record: An AudioRecord instance.
    """

    # Load audio data from the AudioRecord instance.
    data = audio_record.buffer
    if not data.shape[0]:
      # Skip if the audio record's buffer is empty.
      return
    elif len(data) > len(self._buffer):
      # Only get last bits of data that fits in this TensorAudio buffer size.
      data = data[-len(self._buffer), :]

    self.load_from_array(data)

  def load_from_array(self, src: np.ndarray):
    """Load audio data from a NumPy array.

    Args:
      src: A NumPy array contains the input audio.

    Raises:
      ValueError: Raised if the input array has an incorrect shape.
    """
    if len(src) > len(self._buffer):
      raise ValueError('Input audio is too large.')
    elif src.shape[1] != self._format.channels:
      raise ValueError('Input audio contains an invalid number of channels.')

    # Shift the internal buffer backward and add the incoming data to the end of
    # the buffer.
    shift = len(src)
    self._buffer = np.roll(self._buffer, -shift, axis=0)
    self._buffer[-shift:, :] = src

  @property
  def buffer(self):
    return self._buffer


def edgetpu_lib_name():
  """Returns the library name of EdgeTPU in the current platform."""
  return {
      'Darwin': 'libedgetpu.1.dylib',
      'Linux': 'libedgetpu.so.1',
      'Windows': 'edgetpu.dll',
  }.get(platform.system(), None)


class AudioClassifier(object):
  """A wrapper class for the TFLite Audio classification model."""

  def __init__(
      self,
      model_path: str,
      options: AudioClassifierOptions = AudioClassifierOptions()
  ) -> None:
    """Initialize an audio classifier.

    Args:
        model_path: Path of the TFLite audio classification model.
        options: The config for the audio classifier. (Optional)

    Raises:
        ValueError: If the TFLite model is invalid.
        OSError: If the current OS isn't supported by EdgeTPU.
    """
    # Load metadata from model.
    displayer = metadata.MetadataDisplayer.with_model_file(model_path)
    metadata_json = json.loads(displayer.get_metadata_json())
    input_tensor_metadata = metadata_json['subgraph_metadata'][0][
        'input_tensor_metadata'][0]
    input_content_props = input_tensor_metadata['content']['content_properties']
    self._audio_format = AudioFormat(input_content_props['channels'],
                                     input_content_props['sample_rate'])

    # Load label list from metadata.
    file_name = displayer.get_packed_associated_file_list()[0]
    label_map_file = displayer.get_associated_file_buffer(file_name).decode()
    self._labels_list = list(filter(bool, label_map_file.splitlines()))

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

    # Calculate the model input length this way to support both model inputs
    # with and without batch dimension.
    total_input_size = np.prod(interpreter.get_input_details()[0]['shape'])
    self._input_sample_count = int(total_input_size /
                                   self._audio_format.channels)
    self._input_shape = interpreter.get_input_details()[0]['shape']

    self._waveform_input_index = interpreter.get_input_details()[0]['index']
    self._scores_output_index = interpreter.get_output_details()[0]['index']

    self._interpreter = interpreter
    self._options = options

  def create_input_tensor_audio(self) -> TensorAudio:
    """Creates a TensorAudio instance to store the audio input.

    Returns:
        A TensorAudio instance.
    """
    return TensorAudio(
        audio_format=self._audio_format, sample_count=self._input_sample_count)

  def create_audio_record(self) -> AudioRecord:
    """Creates an AudioRecord instance to record audio.

    Returns:
        An AudioRecord instance.
    """
    return AudioRecord(self._audio_format.channels,
                       self._audio_format.sample_rate)

  def classify(self, tensor: TensorAudio) -> List[Category]:
    """Run classification on the input data.

    Args:
        tensor: A TensorAudio instance containing the input audio.

    Returns:
        A list of classification results.
    """
    input_tensor = tensor.buffer.reshape(self._input_shape)
    self._interpreter.allocate_tensors()
    self._interpreter.set_tensor(self._waveform_input_index,
                                 input_tensor.astype(np.float32))
    self._interpreter.invoke()

    scores = self._interpreter.get_tensor(self._scores_output_index)

    return self._postprocess(scores)

  def _postprocess(self, output_tensor: np.ndarray) -> List[Category]:
    """Post-process the output tensor into a list of Category instances.

    Args:
        output_tensor: Output tensor of the TFLite model.

    Returns:
        A list of Category instances.
    """
    # Sort output by score descending.
    scores = np.squeeze(output_tensor)
    score_descending = np.argsort(scores)[::-1]
    categories = [
        Category(label=self._labels_list[idx], score=scores[idx])
        for idx in score_descending
    ]
    filtered_results = categories

    # Filter out classification in deny list.
    if self._options.label_deny_list is not None:
      filtered_results = list(
          filter(
              lambda category: category.label not in self._options.
              label_deny_list, filtered_results))

    # Keep only classification in allow list.
    if self._options.label_allow_list is not None:
      filtered_results = list(
          filter(
              lambda category: category.label in self._options.label_allow_list,
              filtered_results))

    # Filter out classification in score threshold.
    if self._options.score_threshold is not None:
      filtered_results = list(
          filter(
              lambda category: category.score >= self._options.score_threshold,
              filtered_results))

    # Only return maximum of max_results classification.
    if self._options.max_results > 0:
      result_count = min(len(filtered_results), self._options.max_results)
      filtered_results = filtered_results[:result_count]

    return filtered_results
