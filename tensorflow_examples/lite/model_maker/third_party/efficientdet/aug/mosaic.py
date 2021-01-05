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
"""Mosaic augmentation."""
import functools
from typing import Tuple
from absl import logging

import tensorflow as tf


class Mosaic:
  """Mosaic Augmentation class.

  1. Mosaic sub images will not be preserving aspect ratio of original images.
  2. Tested on static graphs and eager  execution.
  3. This Implementation of mosaic augmentation is tested in tf2.x.
  """

  def __init__(self,
               out_size: Tuple(int, int)=(680, 680),
               n_images: int = 4,
               _minimum_mosaic_image_dim: int = 25):
    """__init__.

    Args:
      out_size: output mosaic image size.
      n_images: number images to make mosaic
      _minimum_mosaic_image_dim: minimum percentage of out_size dimension
        should the mosaic be. i.e if out_size is (680,680) and
        _minimum_mosaic_image_dim is 25 , minimum mosaic sub images
        dimension will be 25 % of 680.
    """
    # TODO(someone) #MED #use n_images to build mosaic.
    self._n_images = n_images
    self._out_size = out_size
    self._minimum_mosaic_image_dim = _minimum_mosaic_image_dim
    assert (_minimum_mosaic_image_dim >
            0), "Minimum Mosaic image dimension should be above 0"

  @property
  def n_images(self) -> int:
    return self._n_images

  @property
  def out_size(self) -> int:
    return self._out_size

  def _mosaic_divide_points(self) -> Tuple(int, int):
    """Returns a  tuple of x and y which corresponds to mosaic divide points."""
    x_point = tf.random.uniform(
        shape=[1],
        minval=tf.cast(
            self.out_size[0] * (self._minimum_mosaic_image_dim / 100),
            tf.int32),
        maxval=tf.cast(
            self.out_size[0] * ((100 - self._minimum_mosaic_image_dim) / 100),
            tf.int32,
        ),
        dtype=tf.int32,
    )
    y_point = tf.random.uniform(
        shape=[1],
        minval=tf.cast(
            self.out_size[1] * (self._minimum_mosaic_image_dim / 100),
            tf.int32),
        maxval=tf.cast(
            self.out_size[1] * ((100 - self._minimum_mosaic_image_dim) / 100),
            tf.int32,
        ),
        dtype=tf.int32,
    )
    return x_point, y_point

  @staticmethod
  def _scale_box(box, image, mosaic_image):
    """scale boxes with mosaic sub image.

    Args:
      box: mosaic image box.
      image: original image.
      mosaic_image: mosaic sub image.

    Returns:
      Scaled bounding boxes.
    """
    return [
        box[0] * tf.shape(mosaic_image)[1] / tf.shape(image)[1],
        box[1] * tf.shape(mosaic_image)[0] / tf.shape(image)[0],
        box[2] * tf.shape(mosaic_image)[1] / tf.shape(image)[1],
        box[-1] * tf.shape(mosaic_image)[0] / tf.shape(image)[0],
    ]

  def _scale_images(self, images, mosaic_divide_points):
    """Scale Sub Images.

    Args:
      images: original single images to make mosaic.
      mosaic_divide_points: Points to build mosaic around on given output.

    Returns:
      A tuple of scaled Mosaic sub images.
    """
    x, y = mosaic_divide_points[0][0], mosaic_divide_points[1][0]
    mosaic_image_topleft = tf.image.resize(images[0], (x, y))
    mosaic_image_topright = tf.image.resize(images[1],
                                            (self.out_size[0] - x, y))
    mosaic_image_bottomleft = tf.image.resize(images[2],
                                              (x, self.out_size[1] - y))
    mosaic_image_bottomright = tf.image.resize(
        images[3], (self.out_size[0] - x, self.out_size[1] - y))
    return (
        mosaic_image_topleft,
        mosaic_image_topright,
        mosaic_image_bottomleft,
        mosaic_image_bottomright,
    )

  @tf.function
  def _mosaic(self, images, boxes, mosaic_divide_points):
    """Builds mosaic of provided images.

    Args:
      images: original single images to make mosaic.
      boxes: corresponding bounding boxes to images.
      mosaic_divide_points: Points to build mosaic around on given output size.

    Returns:
      A tuple of mosaic Image, Mosaic Boxes merged.
    """
    (
        mosaic_image_topleft,
        mosaic_image_topright,
        mosaic_image_bottomleft,
        mosaic_image_bottomright,
    ) = self._scale_images(images, mosaic_divide_points)

    #####################################################
    # Scale Boxes for TOP LEFT image.
    # Note: Below function is complex because of TF item assignment restriction.
    # Map_fn is replace with vectorized_map below for optimization purpose.
    mosaic_box_topleft = tf.transpose(
        tf.vectorized_map(
            functools.partial(
                self._scale_box,
                image=images[0],
                mosaic_image=mosaic_image_topleft),
            boxes[0],
        ))

    # Scale and Pad Boxes for TOP RIGHT image.

    mosaic_box_topright = tf.vectorized_map(
        functools.partial(
            self._scale_box,
            image=images[1],
            mosaic_image=mosaic_image_topright),
        boxes[1],
    )
    num_boxes = boxes[1].shape[0]
    idx_tp = tf.constant([[1], [3]])
    update_tp = [
        [tf.shape(mosaic_image_topleft)[0]] * num_boxes,
        [tf.shape(mosaic_image_topleft)[0]] * num_boxes,
    ]
    mosaic_box_topright = tf.transpose(
        tf.tensor_scatter_nd_add(mosaic_box_topright, idx_tp, update_tp))

    # Scale and Pad Boxes for BOTTOM LEFT image.

    mosaic_box_bottomleft = tf.vectorized_map(
        functools.partial(
            self._scale_box,
            image=images[2],
            mosaic_image=mosaic_image_bottomleft),
        boxes[2],
    )

    num_boxes = boxes[2].shape[0]
    idx_bl = tf.constant([[0], [2]])
    update_bl = [
        [tf.shape(mosaic_image_topleft)[1]] * num_boxes,
        [tf.shape(mosaic_image_topleft)[1]] * num_boxes,
    ]
    mosaic_box_bottomleft = tf.transpose(
        tf.tensor_scatter_nd_add(mosaic_box_bottomleft, idx_bl, update_bl))

    # Scale and Pad Boxes for BOTTOM RIGHT image.
    mosaic_box_bottomright = tf.vectorized_map(
        functools.partial(
            self._scale_box,
            image=images[3],
            mosaic_image=mosaic_image_bottomright),
        boxes[3],
    )

    num_boxes = boxes[3].shape[0]
    idx_br = tf.constant([[0], [2], [1], [3]])
    update_br = [
        [tf.shape(mosaic_image_topright)[1]] * num_boxes,
        [tf.shape(mosaic_image_topright)[1]] * num_boxes,
        [tf.shape(mosaic_image_bottomleft)[0]] * num_boxes,
        [tf.shape(mosaic_image_bottomleft)[0]] * num_boxes,
    ]
    mosaic_box_bottomright = tf.transpose(
        tf.tensor_scatter_nd_add(mosaic_box_bottomright, idx_br, update_br))

    # Gather mosaic_sub_images and boxes.
    mosaic_images = [
        mosaic_image_topleft,
        mosaic_image_topright,
        mosaic_image_bottomleft,
        mosaic_image_bottomright,
    ]
    mosaic_boxes = [
        mosaic_box_topleft,
        mosaic_box_topright,
        mosaic_box_bottomleft,
        mosaic_box_bottomright,
    ]

    return mosaic_images, mosaic_boxes

  def __call__(self, images, boxes):
    """Builds mosaic with given images, boxes."""
    if images.shape[0] != 4:
      err_msg = "Currently Exact 4 Images are supported by Mosaic Aug."
      logging.error(err_msg)
      raise Exception(err_msg)

    x, y = self._mosaic_divide_points()
    mosaic_sub_images, mosaic_boxes = self._mosaic(
        images, boxes, mosaic_divide_points=(x, y))

    upper_stack = tf.concat([mosaic_sub_images[0], mosaic_sub_images[1]],
                            axis=0)
    lower_stack = tf.concat([mosaic_sub_images[2], mosaic_sub_images[3]],
                            axis=0)
    mosaic_image = tf.concat([upper_stack, lower_stack], axis=1)
    return mosaic_image, mosaic_boxes
