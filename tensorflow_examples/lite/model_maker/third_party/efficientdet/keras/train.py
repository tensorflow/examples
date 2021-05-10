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
"""The main training script."""
import os
import platform
from absl import app
from absl import flags
from absl import logging
import tensorflow as tf

from tensorflow_examples.lite.model_maker.third_party.efficientdet import dataloader
from tensorflow_examples.lite.model_maker.third_party.efficientdet import hparams_config
from tensorflow_examples.lite.model_maker.third_party.efficientdet import utils
from tensorflow_examples.lite.model_maker.third_party.efficientdet.keras import tfmot
from tensorflow_examples.lite.model_maker.third_party.efficientdet.keras import train_lib
from tensorflow_examples.lite.model_maker.third_party.efficientdet.keras import util_keras

FLAGS = flags.FLAGS


def define_flags():
  """Define the flags."""
  # Cloud TPU Cluster Resolvers
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

  # Model specific paramenters
  flags.DEFINE_string(
      'eval_master',
      default='',
      help='GRPC URL of the eval master. Set to an appropriate value when '
      'running on CPU/GPU')
  flags.DEFINE_string('eval_name', default=None, help='Eval job name')
  flags.DEFINE_enum('strategy', '', ['tpu', 'gpus', ''],
                    'Training: gpus for multi-gpu, if None, use TF default.')

  flags.DEFINE_integer(
      'num_cores', default=8, help='Number of TPU cores for training')

  flags.DEFINE_bool('use_fake_data', False, 'Use fake input.')
  flags.DEFINE_bool(
      'use_xla', False,
      'Use XLA even if strategy is not tpu. If strategy is tpu, always use XLA,'
      ' and this flag has no effect.')
  flags.DEFINE_string('model_dir', None, 'Location of model_dir')

  flags.DEFINE_string('pretrained_ckpt', None,
                      'Start training from this EfficientDet checkpoint.')

  flags.DEFINE_string(
      'hparams', '', 'Comma separated k=v pairs of hyperparameters or a module'
      ' containing attributes to use as hyperparameters.')
  flags.DEFINE_integer('batch_size', 64, 'training batch size')
  flags.DEFINE_integer('eval_samples', 5000, 'The number of samples for '
                       'evaluation.')
  flags.DEFINE_integer('steps_per_execution', 1,
                       'Number of steps per training execution.')
  flags.DEFINE_string(
      'train_file_pattern', None,
      'Glob for train data files (e.g., COCO train - minival set)')
  flags.DEFINE_string('val_file_pattern', None,
                      'Glob for evaluation tfrecords (e.g., COCO val2017 set)')
  flags.DEFINE_string(
      'val_json_file', None,
      'COCO validation JSON containing golden bounding boxes. If None, use the '
      'ground truth from the dataloader. Ignored if testdev_dir is not None.')

  flags.DEFINE_enum('mode', 'traineval', ['train', 'traineval'],
                    'job mode: train, traineval.')
  flags.DEFINE_string(
      'hub_module_url', None, 'TF-Hub path/url to EfficientDet module.'
      'If specified, pretrained_ckpt flag should not be used.')
  flags.DEFINE_integer('num_examples_per_epoch', 120000,
                       'Number of examples in one epoch')
  flags.DEFINE_integer('num_epochs', None, 'Number of epochs for training')
  flags.DEFINE_string('model_name', 'efficientdet-d0', 'Model name.')
  flags.DEFINE_bool('debug', False, 'Enable debug mode')
  flags.DEFINE_integer(
      'tf_random_seed', 111111,
      'Fixed random seed for deterministic execution across runs for debugging.'
  )
  flags.DEFINE_bool('profile', False, 'Enable profile mode')


def setup_model(model, config):
  """Build and compile model."""
  model.build((None, *config.image_size, 3))
  model.compile(
      steps_per_execution=config.steps_per_execution,
      optimizer=train_lib.get_optimizer(config.as_dict()),
      loss={
          train_lib.BoxLoss.__name__:
              train_lib.BoxLoss(
                  config.delta, reduction=tf.keras.losses.Reduction.NONE),
          train_lib.BoxIouLoss.__name__:
              train_lib.BoxIouLoss(
                  config.iou_loss_type,
                  config.min_level,
                  config.max_level,
                  config.num_scales,
                  config.aspect_ratios,
                  config.anchor_scale,
                  config.image_size,
                  reduction=tf.keras.losses.Reduction.NONE),
          train_lib.FocalLoss.__name__:
              train_lib.FocalLoss(
                  config.alpha,
                  config.gamma,
                  label_smoothing=config.label_smoothing,
                  reduction=tf.keras.losses.Reduction.NONE),
          tf.keras.losses.SparseCategoricalCrossentropy.__name__:
              tf.keras.losses.SparseCategoricalCrossentropy(
                  from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
      })
  return model


def init_experimental(config):
  """Serialize train config to model directory."""
  tf.io.gfile.makedirs(config.model_dir)
  config_file = os.path.join(config.model_dir, 'config.yaml')
  if not tf.io.gfile.exists(config_file):
    tf.io.gfile.GFile(config_file, 'w').write(str(config))


def main(_):
  # Parse and override hparams
  config = hparams_config.get_detection_config(FLAGS.model_name)
  config.override(FLAGS.hparams)
  if FLAGS.num_epochs:  # NOTE: remove this flag after updating all docs.
    config.num_epochs = FLAGS.num_epochs

  # Parse image size in case it is in string format.
  config.image_size = utils.parse_image_size(config.image_size)

  if FLAGS.use_xla and FLAGS.strategy != 'tpu':
    tf.config.optimizer.set_jit(True)
    for gpu in tf.config.list_physical_devices('GPU'):
      tf.config.experimental.set_memory_growth(gpu, True)

  if FLAGS.debug:
    tf.debugging.set_log_device_placement(True)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    tf.random.set_seed(FLAGS.tf_random_seed)
    logging.set_verbosity(logging.DEBUG)

  if FLAGS.strategy == 'tpu':
    tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)
    tf.config.experimental_connect_to_cluster(tpu_cluster_resolver)
    tf.tpu.experimental.initialize_tpu_system(tpu_cluster_resolver)
    ds_strategy = tf.distribute.TPUStrategy(tpu_cluster_resolver)
    logging.info('All devices: %s', tf.config.list_logical_devices('TPU'))
  elif FLAGS.strategy == 'gpus':
    gpus = tf.config.list_physical_devices('GPU')
    if FLAGS.batch_size % len(gpus):
      raise ValueError(
          'Batch size divide gpus number must be interger, but got %f' %
          (FLAGS.batch_size / len(gpus)))
    if platform.system() == 'Windows':
      # Windows doesn't support nccl use HierarchicalCopyAllReduce instead
      # TODO(fsx950223): investigate HierarchicalCopyAllReduce performance issue
      cross_device_ops = tf.distribute.HierarchicalCopyAllReduce()
    else:
      cross_device_ops = None
    ds_strategy = tf.distribute.MirroredStrategy(
        cross_device_ops=cross_device_ops)
    logging.info('All devices: %s', gpus)
  else:
    if tf.config.list_physical_devices('GPU'):
      ds_strategy = tf.distribute.OneDeviceStrategy('device:GPU:0')
    else:
      ds_strategy = tf.distribute.OneDeviceStrategy('device:CPU:0')

  steps_per_epoch = FLAGS.num_examples_per_epoch // FLAGS.batch_size
  params = dict(
      profile=FLAGS.profile,
      model_name=FLAGS.model_name,
      steps_per_execution=FLAGS.steps_per_execution,
      model_dir=FLAGS.model_dir,
      steps_per_epoch=steps_per_epoch,
      strategy=FLAGS.strategy,
      batch_size=FLAGS.batch_size,
      tf_random_seed=FLAGS.tf_random_seed,
      debug=FLAGS.debug,
      val_json_file=FLAGS.val_json_file,
      eval_samples=FLAGS.eval_samples,
      num_shards=ds_strategy.num_replicas_in_sync)
  config.override(params, True)
  # set mixed precision policy by keras api.
  precision = utils.get_precision(config.strategy, config.mixed_precision)
  policy = tf.keras.mixed_precision.Policy(precision)
  tf.keras.mixed_precision.set_global_policy(policy)

  def get_dataset(is_training, config):
    file_pattern = (
        FLAGS.train_file_pattern
        if is_training else FLAGS.val_file_pattern)
    if not file_pattern:
      raise ValueError('No matching files.')

    return dataloader.InputReader(
        file_pattern,
        is_training=is_training,
        use_fake_data=FLAGS.use_fake_data,
        max_instances_per_image=config.max_instances_per_image,
        debug=FLAGS.debug)(
            config.as_dict())

  with ds_strategy.scope():
    if config.model_optimizations:
      tfmot.set_config(config.model_optimizations.as_dict())
    if FLAGS.hub_module_url:
      model = train_lib.EfficientDetNetTrainHub(
          config=config, hub_module_url=FLAGS.hub_module_url)
    else:
      model = train_lib.EfficientDetNetTrain(config=config)
    model = setup_model(model, config)
    if FLAGS.debug:
      tf.config.run_functions_eagerly(True)
    if FLAGS.pretrained_ckpt and not FLAGS.hub_module_url:
      ckpt_path = tf.train.latest_checkpoint(FLAGS.pretrained_ckpt)
      util_keras.restore_ckpt(
          model,
          ckpt_path,
          config.moving_average_decay,
          exclude_layers=['class_net'])
    init_experimental(config)
    if 'train' in FLAGS.mode:
      val_dataset = get_dataset(False, config) if 'eval' in FLAGS.mode else None
      model.fit(
          get_dataset(True, config),
          epochs=config.num_epochs,
          steps_per_epoch=steps_per_epoch,
          callbacks=train_lib.get_callbacks(config.as_dict(), val_dataset),
          validation_data=val_dataset,
          validation_steps=(FLAGS.eval_samples // FLAGS.batch_size))
    else:
      # Continuous eval.
      for ckpt in tf.train.checkpoints_iterator(
          FLAGS.model_dir, min_interval_secs=180):
        logging.info('Starting to evaluate.')
        # Terminate eval job when final checkpoint is reached.
        try:
          current_epoch = int(os.path.basename(ckpt).split('-')[1])
        except IndexError:
          current_epoch = 0

        val_dataset = get_dataset(False, config)
        logging.info('start loading model.')
        model.load_weights(tf.train.latest_checkpoint(FLAGS.model_dir))
        logging.info('finish loading model.')
        coco_eval = train_lib.COCOCallback(val_dataset, 1)
        coco_eval.set_model(model)
        eval_results = coco_eval.on_epoch_end(current_epoch)
        logging.info('eval results for %s: %s', ckpt, eval_results)

        try:
          utils.archive_ckpt(eval_results, eval_results['AP'], ckpt)
        except tf.errors.NotFoundError:
          # Checkpoint might be not already deleted by the time eval finished.
          logging.info('Checkpoint %s no longer exists, skipping.', ckpt)

        if current_epoch >= config.num_epochs or not current_epoch:
          logging.info('Eval epoch %d / %d', current_epoch, config.num_epochs)
          break


if __name__ == '__main__':
  define_flags()
  logging.set_verbosity(logging.INFO)
  app.run(main)
