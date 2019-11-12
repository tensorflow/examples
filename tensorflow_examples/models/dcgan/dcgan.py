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
"""DCGAN.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
from absl import app
from absl import flags
import tensorflow as tf # TF2
import tensorflow_datasets as tfds
assert tf.__version__.startswith('2')

FLAGS = flags.FLAGS

flags.DEFINE_integer('buffer_size', 10000, 'Shuffle buffer size')
flags.DEFINE_integer('batch_size', 64, 'Batch Size')
flags.DEFINE_integer('epochs', 1, 'Number of epochs')
flags.DEFINE_boolean('enable_function', True, 'Enable Function?')

AUTOTUNE = tf.data.experimental.AUTOTUNE


def scale(image, label):
  image = tf.cast(image, tf.float32)
  image = (image - 127.5) / 127.5

  return image, label


def create_dataset(buffer_size, batch_size):
  train_dataset = tfds.load(
      'mnist', split='train', as_supervised=True, shuffle_files=True)
  train_dataset = train_dataset.map(scale, num_parallel_calls=AUTOTUNE)
  train_dataset = train_dataset.shuffle(buffer_size).batch(batch_size)
  return train_dataset


def make_generator_model():
  """Generator.

  Returns:
    Keras Sequential model
  """
  model = tf.keras.Sequential([
      tf.keras.layers.Dense(7*7*256, use_bias=False),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.LeakyReLU(),
      tf.keras.layers.Reshape((7, 7, 256)),
      tf.keras.layers.Conv2DTranspose(128, 5, strides=(1, 1),
                                      padding='same', use_bias=False),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.LeakyReLU(),
      tf.keras.layers.Conv2DTranspose(64, 5, strides=(2, 2),
                                      padding='same', use_bias=False),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.LeakyReLU(),
      tf.keras.layers.Conv2DTranspose(1, 5, strides=(2, 2),
                                      padding='same', use_bias=False,
                                      activation='tanh')
  ])

  return model


def make_discriminator_model():
  """Discriminator.

  Returns:
    Keras Sequential model
  """
  model = tf.keras.Sequential([
      tf.keras.layers.Conv2D(64, 5, strides=(2, 2), padding='same'),
      tf.keras.layers.LeakyReLU(),
      tf.keras.layers.Dropout(0.3),
      tf.keras.layers.Conv2D(128, 5, strides=(2, 2), padding='same'),
      tf.keras.layers.LeakyReLU(),
      tf.keras.layers.Dropout(0.3),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(1)
  ])

  return model


def get_checkpoint_prefix():
  checkpoint_dir = './training_checkpoints'
  checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')

  return checkpoint_prefix


class Dcgan(object):
  """Dcgan class.

  Args:
    epochs: Number of epochs.
    enable_function: If true, train step is decorated with tf.function.
    batch_size: Batch size.
  """

  def __init__(self, epochs, enable_function, batch_size):
    self.epochs = epochs
    self.enable_function = enable_function
    self.batch_size = batch_size
    self.noise_dim = 100
    self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
    self.generator = make_generator_model()
    self.discriminator = make_discriminator_model()
    self.checkpoint = tf.train.Checkpoint(
        generator_optimizer=self.generator_optimizer,
        discriminator_optimizer=self.discriminator_optimizer,
        generator=self.generator,
        discriminator=self.discriminator)

  def generator_loss(self, generated_output):
    return self.loss_object(tf.ones_like(generated_output), generated_output)

  def discriminator_loss(self, real_output, generated_output):
    real_loss = self.loss_object(tf.ones_like(real_output), real_output)
    generated_loss = self.loss_object(
        tf.zeros_like(generated_output), generated_output)

    total_loss = real_loss + generated_loss

    return total_loss

  def train_step(self, image):
    """One train step over the generator and discriminator model.

    Args:
      image: Input image.

    Returns:
      generator loss, discriminator loss.
    """
    noise = tf.random.normal([self.batch_size, self.noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = self.generator(noise, training=True)

      real_output = self.discriminator(image, training=True)
      generated_output = self.discriminator(generated_images, training=True)

      gen_loss = self.generator_loss(generated_output)
      disc_loss = self.discriminator_loss(real_output, generated_output)

    gradients_of_generator = gen_tape.gradient(
        gen_loss, self.generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(
        disc_loss, self.discriminator.trainable_variables)

    self.generator_optimizer.apply_gradients(zip(
        gradients_of_generator, self.generator.trainable_variables))
    self.discriminator_optimizer.apply_gradients(zip(
        gradients_of_discriminator, self.discriminator.trainable_variables))

    return gen_loss, disc_loss

  def train(self, dataset, checkpoint_pr):
    """Train the GAN for x number of epochs.

    Args:
      dataset: train dataset.
      checkpoint_pr: prefix in which the checkpoints are stored.

    Returns:
      Time for each epoch.
    """
    time_list = []
    if self.enable_function:
      self.train_step = tf.function(self.train_step)

    for epoch in range(self.epochs):
      start_time = time.time()
      for image, _ in dataset:
        gen_loss, disc_loss = self.train_step(image)

      wall_time_sec = time.time() - start_time
      time_list.append(wall_time_sec)

      # saving (checkpoint) the model every 15 epochs
      if (epoch + 1) % 15 == 0:
        self.checkpoint.save(file_prefix=checkpoint_pr)

      template = 'Epoch {}, Generator loss {}, Discriminator Loss {}'
      print (template.format(epoch, gen_loss, disc_loss))

    return time_list


def run_main(argv):
  del argv
  kwargs = {'epochs': FLAGS.epochs, 'enable_function': FLAGS.enable_function,
            'buffer_size': FLAGS.buffer_size, 'batch_size': FLAGS.batch_size}
  main(**kwargs)


def main(epochs, enable_function, buffer_size, batch_size):
  train_dataset = create_dataset(buffer_size, batch_size)
  checkpoint_pr = get_checkpoint_prefix()

  dcgan_obj = Dcgan(epochs, enable_function, batch_size)
  print ('Training ...')
  return dcgan_obj.train(train_dataset, checkpoint_pr)

if __name__ == '__main__':
  app.run(run_main)
