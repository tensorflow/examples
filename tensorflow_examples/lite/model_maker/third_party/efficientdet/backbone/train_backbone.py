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
import re

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow_examples.lite.model_maker.third_party.efficientdet.backbone import efficientnet_builder
from tensorflow_examples.lite.model_maker.third_party.efficientdet.backbone import efficientnet_model
from tensorflow_examples.lite.model_maker.third_party.efficientdet.backbone import preprocessing

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
flags.DEFINE_float(
    'label_smoothing', 0.1,
    'Label smoothing parameter used in the softmax_cross_entropy')
flags.DEFINE_float('weight_decay', 1e-5,
                   'Weight decay coefficiant for l2 regularization.')
flags.DEFINE_string(
    'augment_name', None, '`string` that is the name of the augmentation method'
    'to apply to the image. `autoaugment` if AutoAugment is to be used or'
    '`randaugment` if RandAugment is to be used. If the value is `None` no'
    'augmentation method will be applied applied. See autoaugment.py for  '
    'more details.')

flags.DEFINE_integer(
    'randaug_num_layers', 2,
    'If RandAug is used, what should the number of layers be.'
    'See autoaugment.py for detailed description.')

flags.DEFINE_integer(
    'randaug_magnitude', 10,
    'If RandAug is used, what should the magnitude be. '
    'See autoaugment.py for detailed description.')

FLAGS = flags.FLAGS


def create_dataset(dataset: tf.data.Dataset, num_classes: int,
                   is_training: bool) -> tf.data.Dataset:
  """Produces a full, augmented dataset from the inptu dataset."""
  _, _, resolution, _ = efficientnet_builder.efficientnet_params(
      FLAGS.model_name)

  def process_data(image, label):
    image = preprocessing.preprocess_image(
        image,
        is_training=is_training,
        use_bfloat16=FLAGS.strategy == 'tpus',
        image_size=resolution,
        augment_name=FLAGS.augment_name,
        randaug_num_layers=FLAGS.randaug_num_layers,
        randaug_magnitude=FLAGS.randaug_magnitude,
        resize_method=None)

    label = tf.one_hot(label, num_classes)
    return image, label

  dataset = dataset.map(
      process_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  dataset = dataset.batch(FLAGS.batch_size, drop_remainder=True)
  dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

  return dataset


class TrainableModel(efficientnet_model.Model):
  """Wraps efficientnet to make a keras trainable model.

  Handles efficientnet's multiple outputs and adds weight decay.
  """

  def __init__(self,
               blocks_args=None,
               global_params=None,
               name=None,
               weight_decay=0.0):
    super().__init__(
        blocks_args=blocks_args, global_params=global_params, name=name)

    self.weight_decay = weight_decay

  def _reg_l2_loss(self, weight_decay, regex=r'.*(kernel|weight):0$'):
    """Return regularization l2 loss loss."""
    var_match = re.compile(regex)
    return weight_decay * tf.add_n([
        tf.nn.l2_loss(v)
        for v in self.trainable_variables
        if var_match.match(v.name)
    ])

  def train_step(self, data):
    images, labels = data

    with tf.GradientTape() as tape:
      pred = self(images, training=True)[0]
      loss = self.compiled_loss(
          labels,
          pred,
          regularization_losses=[self._reg_l2_loss(self.weight_decay)])

    self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
    self.compiled_metrics.update_state(labels, pred)
    return {m.name: m.result() for m in self.metrics}

  def test_step(self, data):
    images, labels = data
    pred = self(images, training=False)[0]

    self.compiled_loss(
        labels,
        pred,
        regularization_losses=[self._reg_l2_loss(self.weight_decay)])

    self.compiled_metrics.update_state(labels, pred)
    return {m.name: m.result() for m in self.metrics}


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
        decoders={
            'image': tfds.decode.SkipDecoding(),
        },
    )

    num_classes = ds_info.features['label'].num_classes

    ds_train = create_dataset(ds_train, num_classes, is_training=True)

    ds_val = create_dataset(ds_val, num_classes, is_training=False)

    blocks_args, global_params = efficientnet_builder.get_model_params(
        FLAGS.model_name,
        override_params={'num_classes': ds_info.features['label'].num_classes})
    model = TrainableModel(blocks_args, global_params, FLAGS.model_name,
                           FLAGS.weight_decay)

    steps_per_epoch = ds_info.splits['train'].num_examples // FLAGS.batch_size
    total_steps = steps_per_epoch * FLAGS.num_epochs
    cosign_decay = tf.keras.experimental.CosineDecay(FLAGS.learning_rate,
                                                     total_steps)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(cosign_decay),
        loss=tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=FLAGS.label_smoothing, from_logits=True),
        metrics=[tf.keras.metrics.CategoricalAccuracy()],
    )

    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(FLAGS.model_dir, 'ckpt-{epoch:d}'),
        verbose=1,
        save_weights_only=True)
    tb_callback = tf.keras.callbacks.TensorBoard(
        log_dir=FLAGS.model_dir, update_freq=100)
    rstr_callback = tf.keras.callbacks.experimental.BackupAndRestore(
        backup_dir=FLAGS.model_dir)

    model.fit(
        ds_train,
        epochs=FLAGS.num_epochs,
        validation_data=ds_val,
        callbacks=[ckpt_callback, tb_callback, rstr_callback],
        # don't log spam if running on tpus
        verbose=2 if FLAGS.strategy == 'tpu' else 1,
    )


if __name__ == '__main__':
  app.run(main)
