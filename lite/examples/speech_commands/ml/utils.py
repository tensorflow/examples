# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def data_gen(audio_processor,
             sess,
             batch_size=128,
             background_frequency=0.3,
             background_volume_range=0.15,
             foreground_frequency=0.3,
             foreground_volume_range=0.15,
             time_shift_frequency=0.3,
             time_shift_range=[-500, 0],
             mode='validation',
             flip_frequency=0.0,
             silence_volume_range=0.3):
  ep_count = 0
  offset = 0
  if mode != 'training':
    background_frequency = 0.0
    background_volume_range = 0.0
    foreground_frequency = 0.0
    foreground_volume_range = 0.0
    time_shift_frequency = 0.0
    time_shift_range = [0, 0]
    flip_frequency = 0.0
    # silence_volume_range: stays the same for validation
  while True:
    X, y = audio_processor.get_data(
        how_many=batch_size,
        offset=0 if mode == 'training' else offset,
        background_frequency=background_frequency,
        background_volume_range=background_volume_range,
        foreground_frequency=foreground_frequency,
        foreground_volume_range=foreground_volume_range,
        time_shift_frequency=time_shift_frequency,
        time_shift_range=time_shift_range,
        mode=mode,
        sess=sess,
        flip_frequency=flip_frequency,
        silence_volume_range=silence_volume_range)
    offset += batch_size
    if offset > audio_processor.set_size(mode) - batch_size:
      offset = 0
      print('\n[Ep:%03d: %s-mode]' % (ep_count, mode))
      ep_count += 1
    yield X, y


def tf_roll(a, shift, a_len=16000):
  # https://stackoverflow.com/questions/42651714/vector-shift-roll-in-tensorflow
  def roll_left(a, shift, a_len):
    shift %= a_len
    rolled = tf.concat([a[a_len - shift:, :], a[:a_len - shift, :]], axis=0)
    return rolled

  def roll_right(a, shift, a_len):
    shift = -shift
    shift %= a_len
    rolled = tf.concat([a[shift:, :], a[:shift, :]], axis=0)
    return rolled

  # https://stackoverflow.com/questions/35833011/how-to-add-if-condition-in-a-tensorflow-graph
  return tf.cond(
      tf.greater_equal(shift, 0),
      true_fn=lambda: roll_left(a, shift, a_len),
      false_fn=lambda: roll_right(a, shift, a_len))
