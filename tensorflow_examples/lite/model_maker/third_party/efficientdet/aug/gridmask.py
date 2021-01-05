# Copyright 2020 Google Research. All Rights Reserved.
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
"""Grid Masking Augmentation Reference: https://arxiv.org/abs/2001.04086."""
import math

import tensorflow as tf
from tensorflow_addons import image as image_ops


class GridMask(object):
  """GridMask class for grid masking augmentation."""

  def __init__(self,
               prob=0.6,
               ratio=0.6,
               rotate=10,
               gridmask_size_ratio=0.5,
               fill=1,
               interpolation="BILINEAR"):
    """initialization.

    Args:
      prob: probablity of occurance.
      ratio: grid mask ratio i.e if 0.5 grid and spacing will be equal.
      rotate: Rotation of grid mesh.
      gridmask_size_ratio: Grid mask size, grid to image size ratio.
      fill: Fill value for grids.
      interpolation: Interpolation method for rotation.
    """
    self.prob = prob
    self.ratio = ratio
    self.rotate = rotate
    self.gridmask_size_ratio = gridmask_size_ratio
    self.fill = fill
    self.interpolation = interpolation

  @tf.function
  def random_rotate(self, mask):
    """Randomly rotates mask on given range."""

    angle = self.rotate * tf.random.normal([], -1, 1)
    angle = math.pi * angle / 180
    return image_ops.rotate(mask, angle, interpolation=self.interpolation)

  @staticmethod
  def crop(mask, h, w):
    """crops in middle of mask and image corners."""
    ww = hh = tf.shape(mask)[0]
    mask = mask[(hh - h) // 2:(hh - h) // 2 + h,
                (ww - w) // 2:(ww - w) // 2 + w,]
    return mask

  @tf.function
  def mask(self, h, w):
    """mask helper function for initializing grid mask of required size."""
    h = tf.cast(h, tf.float32)
    w = tf.cast(w, tf.float32)
    mask_w = mask_h = tf.cast(
        tf.cast(
            (self.gridmask_size_ratio + 1), tf.float32) * tf.math.maximum(h, w),
        tf.int32)
    self.mask_w = mask_w
    mask = tf.zeros(shape=[mask_h, mask_w], dtype=tf.int32)
    gridblock = tf.random.uniform(
        shape=[],
        minval=int(tf.math.minimum(h * 0.5, w * 0.3)),
        maxval=int(tf.math.maximum(h * 0.5, w * 0.3)) + 1,
        dtype=tf.int32)

    if self.ratio == 1:
      length = tf.random.uniform(
          shape=[], minval=1, maxval=gridblock + 1, dtype=tf.int32)
    else:
      length = tf.cast(
          tf.math.minimum(
              tf.math.maximum(
                  int(tf.cast(gridblock, tf.float32) * self.ratio + 0.5), 1),
              gridblock - 1), tf.int32)

    for _ in range(2):
      start_w = tf.random.uniform(
          shape=[], minval=0, maxval=gridblock + 1, dtype=tf.int32)
      for i in range(mask_w // gridblock):
        start = gridblock * i + start_w
        end = tf.math.minimum(start + length, mask_w)
        indices = tf.reshape(tf.range(start, end), [end - start, 1])
        updates = (
            tf.ones(shape=[end - start, mask_w], dtype=tf.int32) * self.fill)
        mask = tf.tensor_scatter_nd_update(mask, indices, updates)
      mask = tf.transpose(mask)

    return mask

  def __call__(self, image, label):
    """Masks input image tensor with random grid mask."""
    h = tf.shape(image)[0]
    w = tf.shape(image)[1]
    grid = self.mask(h, w)
    grid = self.random_rotate(grid)
    mask = self.crop(grid, h, w)
    mask = tf.cast(mask, image.dtype)
    mask = tf.reshape(mask, (h, w))
    mask = (tf.expand_dims(mask, -1) if image._rank() != mask._rank() else mask)
    occur = tf.random.normal([], 0, 1) < self.prob
    image = tf.cond(occur, lambda: image * mask, lambda: image)
    return image, label


def gridmask(image,
             boxes,
             prob=0.5,
             ratio=0.6,
             rotate=10,
             gridmask_size_ratio=0.5,
             fill=1):
  """Callable instance of GridMask and transforms input image."""
  gridmask_obj = GridMask(
      prob=prob,
      ratio=ratio,
      rotate=rotate,
      gridmask_size_ratio=gridmask_size_ratio,
      fill=fill)
  image, boxes = gridmask_obj(image, boxes)
  return image, boxes
