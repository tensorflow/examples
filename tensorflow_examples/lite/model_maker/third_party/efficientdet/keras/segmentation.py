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
"""A demo script to show to train a segmentation model."""
from absl import app
from absl import logging
import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow_examples.lite.model_maker.third_party.efficientdet import hparams_config
from tensorflow_examples.lite.model_maker.third_party.efficientdet.keras import efficientdet_keras


def create_mask(pred_mask):
  pred_mask = tf.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask[0]


def normalize(input_image, input_mask):
  input_image = tf.cast(input_image, tf.float32) / 255.0
  input_mask -= 1
  return input_image, input_mask


def load_image_train(datapoint):
  """Load images for training."""
  input_image = tf.image.resize(datapoint['image'], (512, 512))
  input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

  if tf.random.uniform(()) > 0.5:
    input_image = tf.image.flip_left_right(input_image)
    input_mask = tf.image.flip_left_right(input_mask)

  input_image, input_mask = normalize(input_image, input_mask)

  return input_image, input_mask


def load_image_test(datapoint):
  input_image = tf.image.resize(datapoint['image'], (512, 512))
  input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

  input_image, input_mask = normalize(input_image, input_mask)

  return input_image, input_mask


def main(_):
  dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)
  train_examples = info.splits['train'].num_examples
  batch_size = 8
  steps_per_epoch = train_examples // batch_size

  train = dataset['train'].map(
      load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  test = dataset['test'].map(load_image_test)

  train_dataset = train.cache().shuffle(1000).batch(batch_size).repeat()
  train_dataset = train_dataset.prefetch(
      buffer_size=tf.data.experimental.AUTOTUNE)
  test_dataset = test.batch(batch_size)
  config = hparams_config.get_efficientdet_config('efficientdet-d0')
  config.heads = ['segmentation']
  model = efficientdet_keras.EfficientDetNet(config=config)
  model.build((1, 512, 512, 3))
  model.compile(
      optimizer='adam',
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=['accuracy'])

  val_subsplits = 5
  val_steps = info.splits['test'].num_examples // batch_size // val_subsplits
  model.fit(
      train_dataset,
      epochs=20,
      steps_per_epoch=steps_per_epoch,
      validation_steps=val_steps,
      validation_data=test_dataset,
      callbacks=[])

  model.save_weights('./testdata/segmentation')

  print(create_mask(model(tf.ones((1, 512, 512, 3)), False)))


if __name__ == '__main__':
  logging.set_verbosity(logging.WARNING)
  app.run(main)
