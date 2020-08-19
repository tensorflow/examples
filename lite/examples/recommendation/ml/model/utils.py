# Lint as: python3
#   Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""Utilities for ondevice personalized recommendations model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Iterable, Tuple, TypeVar

import tensorflow as tf

TFGradient = TypeVar('TFGradient', tf.Tensor, tf.IndexedSlices)

Scalar = TypeVar('Scalar', tf.Variable, tf.Tensor, float, int)


def GetShardFilenames(filepattern):
  """Get a list of filenames given a pattern.

  This will also check whether the files exist on the filesystem. The pattern
  can be either of the glob form, or the 'basename@num_shards' form.

  Args:
    filepattern: File pattern.

  Returns:
    A list of shard patterns.

  Raises:
    ValueError: if using the shard pattern, if some shards don't exist.
  """
  filenames = tf.io.gfile.glob(filepattern)
  for filename in filenames:
    if not tf.io.gfile.exists(filename):
      raise ValueError('File not found: %s' % filename)
  return filenames


def ClipGradient(
    grads_and_vars: Iterable[Tuple[TFGradient, tf.Variable]],
    clip_val: Scalar = 1.0,
    include_histogram_summary: bool = False
) -> Tuple[Tuple[TFGradient, tf.Variable], ...]:
  """Clips all gradients by global norm, reducing norm to clip_val.

  Args:
    grads_and_vars: Gradients and vars list input.
    clip_val: A 0-D (scalar) `Tensor` > 0. Value to clip to.
    include_histogram_summary: Flag indicates adding clipped gradients to
      histogram summary.

  Returns:
    clipped_grads_and_vars: Return gradients and vars list after operation.
  """

  grads, variables = list(zip(*grads_and_vars))
  with tf.name_scope('gradient_clip'):
    clipped_grads, global_norm = tf.clip_by_global_norm(grads, clip_val)
    if include_histogram_summary:
      tf.summary.scalar('clipped_global_norm', global_norm)

  clipped_grads_and_vars = tuple(zip(clipped_grads, variables))

  if include_histogram_summary:
    with tf.name_scope('Gradient_info_clip'):
      for g, v in clipped_grads_and_vars:
        if not isinstance(g, tf.IndexedSlices):
          tf.summary.scalar('%s_clip_l2_norm' % v.name.replace(':', '_'),
                            tf.sqrt(tf.reduce_sum(tf.square(g))))
  return clipped_grads_and_vars
