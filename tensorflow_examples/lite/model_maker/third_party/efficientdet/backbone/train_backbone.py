# Copyright 2021 Google Research. All Rights Reserved.
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
"""A simple script to allow pretraining of the efficientnet backbone."""

import os

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow_examples.lite.model_maker.third_party.efficientdet.backbone import efficientnet_builder

flags.DEFINE_string(
    'tpu',
    default=None,
    help='The Cloud TPU to use for training. This should be either the name '
    'used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 '
    'url.')
flags.DEFINE_string(
    'gcp_project',
    default=None,
    help='Project name for the Cloud TPU-enabled project. If not specified, '
    'we will attempt to automatically detect the GCE project from metadata.')
flags.DEFINE_string(
    'tpu_zone',
    default=None,
    help='GCE zone where the Cloud TPU is located in. If not specified, we '
    'will attempt to automatically detect the GCE project from metadata.')

flags.DEFINE_enum('strategy', '', ['tpu', 'gpus', ''],
                  'Training: gpus for multi-gpu, if None, use TF default.')
flags.DEFINE_string(
    'model_dir', None, 'The directory where the model weights and '
    'training/evaluation summaries are stored.')
flags.DEFINE_string('model_name', 'efficientnet-b0', 'Model name.')
flags.DEFINE_integer('batch_size', 64, 'training batch size')
flags.DEFINE_integer('num_epochs', 10, 'Number of epochs to train.')
flags.DEFINE_float('learning_rate', 1e-4, 'The learning rate to use.')

FLAGS = flags.FLAGS


def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  _, _, resolution, _ = efficientnet_builder.efficientnet_params(
      FLAGS.model_name)

  image = tf.cast(image, tf.float32) / 255.
  image = tf.image.resize(image, (resolution, resolution))
  return image, label


def create_dataset(dataset: tf.data.Dataset) -> tf.data.Dataset:
  dataset = dataset.map(
      normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  dataset = dataset.batch(FLAGS.batch_size, drop_remainder=True)
  dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

  return dataset


def main(_) -> None:
  if FLAGS.strategy == 'tpu':
    tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)
    tf.config.experimental_connect_to_cluster(tpu_cluster_resolver)
    tf.tpu.experimental.initialize_tpu_system(tpu_cluster_resolver)
    ds_strategy = tf.distribute.TPUStrategy(tpu_cluster_resolver)
    logging.info('All devices: %s', tf.config.list_logical_devices('TPU'))
  elif FLAGS.strategy == 'gpus':
    ds_strategy = tf.distribute.MirroredStrategy()
    logging.info('All devices: %s', tf.config.list_physical_devices('GPU'))
  else:
    if tf.config.list_physical_devices('GPU'):
      ds_strategy = tf.distribute.OneDeviceStrategy('device:GPU:0')
    else:
      ds_strategy = tf.distribute.OneDeviceStrategy('device:CPU:0')

  with ds_strategy.scope():
    (ds_train, ds_val), ds_info = tfds.load(
        'imagenet2012',
        split=['train', 'validation'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    ds_train = create_dataset(ds_train)

    ds_val = create_dataset(ds_val)

    _, _, resolution, _ = efficientnet_builder.efficientnet_params(
        FLAGS.model_name)

    inputs = tf.keras.Input(
        shape=(resolution, resolution, 3), batch_size=FLAGS.batch_size)

    outputs = efficientnet_builder.build_model(
        inputs,
        FLAGS.model_name,
        training=True,
        override_params={'num_classes': ds_info.features['label'].num_classes})

    model = tf.keras.Model(inputs=inputs, outputs=outputs[0])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(FLAGS.learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(FLAGS.model_dir, 'ckpt-{epoch:d}'),
        verbose=1,
        save_weights_only=True)
    tb_callback = tf.keras.callbacks.TensorBoard(
        log_dir=FLAGS.model_dir, update_freq=100)

    model.fit(
        ds_train,
        epochs=FLAGS.num_epochs,
        validation_data=ds_val,
        callbacks=[ckpt_callback, tb_callback],
    )


if __name__ == '__main__':
  app.run(main)
