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
"""A module with util functions."""
import sys
from typing import List

from audio_classifier import Category
from matplotlib import rcParams
import matplotlib.pyplot as plt

rcParams.update({
    # Set the plot left margin so that the labels are visible.
    'figure.subplot.left': 0.3,

    # Hide the bottom toolbar.
    'toolbar': 'None'
})


class Plotter(object):
  """An util class to display the classification results."""

  _PAUSE_TIME = 0.05
  """Time for matplotlib to wait for UI event."""

  def __init__(self) -> None:
    fig, self._axes = plt.subplots()
    fig.canvas.manager.set_window_title('Audio classification')

    # Stop the program when the ESC key is pressed.
    def event_callback(event):
      if event.key == 'escape':
        sys.exit(0)

    fig.canvas.mpl_connect('key_press_event', event_callback)

    plt.show(block=False)

  def plot(self, categories: List[Category]) -> None:
    """Plot the audio classification result.

    Args:
        categories: A list of Category instances from an audio classification
          model.
    """
    # Clear the axes
    self._axes.cla()
    self._axes.set_title('Press ESC to exit.')
    self._axes.set_xlim((0, 1))

    # Plot the results so that the most probable category comes at the top.
    label_list = [category.label for category in categories]
    score_list = [category.score for category in categories]
    self._axes.barh(label_list[::-1], score_list[::-1])

    # Wait for the UI event.
    plt.pause(self._PAUSE_TIME)
