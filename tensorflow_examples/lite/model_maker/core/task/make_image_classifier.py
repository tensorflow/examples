# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Utils for building image classifiers."""

import collections
import tensorflow as tf


class HParams(
    collections.namedtuple(
        'HParams',
        [
            'train_epochs',
            'do_fine_tuning',
            'batch_size',
            'learning_rate',
            'momentum',
            'dropout_rate',
            'l1_regularizer',
            'l2_regularizer',
            'label_smoothing',
            'validation_split',
            'do_data_augmentation',
            'rotation_range',
            'horizontal_flip',
            'width_shift_range',
            'height_shift_range',
            'shear_range',
            'zoom_range',
        ],
    )
):
  """The hyperparameters for make_image_classifier.

  train_epochs: Training will do this many iterations over the dataset.
  do_fine_tuning: If true, the Hub module is trained together with the
    classification layer on top.
  batch_size: Each training step samples a batch of this many images.
  learning_rate: The learning rate to use for gradient descent training.
  momentum: The momentum parameter to use for gradient descent training.
  dropout_rate: The fraction of the input units to drop, used in dropout layer.
  """


def get_default_hparams():
  return HParams(
      train_epochs=5,
      do_fine_tuning=False,
      batch_size=32,
      learning_rate=0.005,
      momentum=0.9,
      dropout_rate=0.2,
      l1_regularizer=0.0,
      l2_regularizer=0.0001,
      label_smoothing=0.1,
      validation_split=0.2,
      do_data_augmentation=False,
      rotation_range=40,
      horizontal_flip=True,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0,
      zoom_range=0.2,
  )


def build_model(module_layer, hparams, image_size, num_classes):
  """Builds the full classifier model from the given module_layer.

  If using a DistributionStrategy, call this under its `.scope()`.
  Args:
    module_layer: Pre-trained tfhub model layer.
    hparams: A namedtuple of hyperparameters. This function expects
      .dropout_rate: The fraction of the input units to drop, used in dropout
      layer.
    image_size: The input image size to use with the given module layer.
    num_classes: Number of the classes to be predicted.

  Returns:
    The full classifier model.
  """
  model = tf.keras.Sequential([
      tf.keras.Input(shape=(image_size[0], image_size[1], 3)),
      module_layer,
      tf.keras.layers.Dropout(rate=hparams.dropout_rate),
      tf.keras.layers.Dense(
          num_classes,
          activation='softmax',
          kernel_regularizer=tf.keras.regularizers.l1_l2(
              l1=hparams.l1_regularizer, l2=hparams.l2_regularizer
          ),
      ),
  ])
  print(model.summary())
  return model
