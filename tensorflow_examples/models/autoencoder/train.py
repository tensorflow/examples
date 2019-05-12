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
"""Training module for Vanilla Autoencoder"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from autoencoder import Autoencoder
from absl import app
from absl import flags
import numpy as np
import os
import time
import tensorflow as tf
from tensorflow.keras.datasets import mnist
assert tf.__version__.startswith('2')

tf.config.gpu.set_per_process_memory_growth(True)
tf.random.set_seed(42)
np.random.seed(42)

FLAGS = flags.FLAGS

flags.DEFINE_integer('buffer_size', 10000, 'Shuffle buffer size')
flags.DEFINE_integer('batch_size', 128, 'Batch size')
flags.DEFINE_integer('epochs', 10, 'Number of epochs')


def scale(image):
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [-1, 784])
    image = image / 255.
    return image


def create_dataset(features, buffer_size, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((features))
    dataset = dataset.map(scale)
    dataset = dataset.prefetch((features.shape[0] // batch_size)).shuffle(buffer_size).batch(batch_size, drop_remainder=True)
    return dataset


def run_main(argv):
    kwargs = {'epochs': FLAGS.epochs, 'buffer_size': FLAGS.buffer_size,
              'batch_size': FLAGS.batch_size}
    main(**kwargs)


def main(epochs, buffer_size, batch_size):
    (train_features, _), _ = mnist.load_data()
    dataset = create_dataset(train_features, buffer_size, batch_size)

    model = Autoencoder(latent_dim=32, original_dim=784, units=64)
    model.train(dataset, epochs=epochs)


if __name__ == '__main__':
    app.run(run_main)

