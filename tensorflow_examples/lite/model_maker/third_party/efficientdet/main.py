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
import multiprocessing
import os
from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf

from tensorflow_examples.lite.model_maker.third_party.efficientdet import dataloader
from tensorflow_examples.lite.model_maker.third_party.efficientdet import det_model_fn
from tensorflow_examples.lite.model_maker.third_party.efficientdet import hparams_config
from tensorflow_examples.lite.model_maker.third_party.efficientdet import utils

flags.DEFINE_string(
    'tpu',
    default=None,
    help='The Cloud TPU to use for training. This should be either the name '
    'used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 '
    'url.')
flags.DEFINE_string(
    'gcp_project',
    default=None,
    help='Project name for the Cloud TPU-enabled project. If not specified, we '
    'will attempt to automatically detect the GCE project from metadata.')
flags.DEFINE_string(
    'tpu_zone',
    default=None,
    help='GCE zone where the Cloud TPU is located in. If not specified, we '
    'will attempt to automatically detect the GCE project from metadata.')
flags.DEFINE_string('eval_name', default=None, help='Eval job name')
flags.DEFINE_enum('strategy', None, ['tpu', 'gpus', ''],
                  'Training: gpus for multi-gpu, if None, use TF default.')

flags.DEFINE_bool('use_fake_data', False, 'Use fake input.')
flags.DEFINE_bool(
    'use_xla', False,
    'Use XLA even if strategy is not tpu. If strategy is tpu, always use XLA, '
    'and this flag has no effect.')
flags.DEFINE_string('model_dir', None, 'Location of model_dir')
flags.DEFINE_string(
    'backbone_ckpt', '', 'Location of the ResNet50 checkpoint to use for model '
    'initialization.')
flags.DEFINE_string('ckpt', None,
                    'Start training from this EfficientDet checkpoint.')

flags.DEFINE_string(
    'hparams', '', 'Comma separated k=v pairs of hyperparameters or a module'
    ' containing attributes to use as hyperparameters.')
flags.DEFINE_integer(
    'num_cores', default=8, help='Number of TPU cores for training')
flags.DEFINE_bool('use_spatial_partition', False, 'Use spatial partition.')
flags.DEFINE_integer(
    'num_cores_per_replica',
    default=2,
    help='Number of TPU cores per replica when using spatial partition.')
flags.DEFINE_multi_integer(
    'input_partition_dims', [1, 2, 1, 1],
    'A list that describes the partition dims for all the tensors.')
flags.DEFINE_integer('train_batch_size', 64, 'global training batch size')
flags.DEFINE_integer('eval_batch_size', 1, 'global evaluation batch size')
flags.DEFINE_integer('eval_samples', 5000, 'Number of samples for eval.')
flags.DEFINE_integer('iterations_per_loop', 100,
                     'Number of iterations per TPU training loop')
flags.DEFINE_integer('save_checkpoints_steps', 100,
                     'Number of iterations per checkpoint save')
flags.DEFINE_string(
    'train_file_pattern', None,
    'Glob for training data files (e.g., COCO train - minival set)')
flags.DEFINE_string('val_file_pattern', None,
                    'Glob for evaluation tfrecords (e.g., COCO val2017 set)')
flags.DEFINE_string(
    'val_json_file', None,
    'COCO validation JSON containing golden bounding boxes. If None, use the '
    'ground truth from the dataloader. Ignored if testdev_dir is not None.')
flags.DEFINE_string('testdev_dir', None,
                    'COCO testdev dir. If not None, ignorer val_json_file.')
flags.DEFINE_integer('num_examples_per_epoch', 120000,
                     'Number of examples in one epoch')
flags.DEFINE_integer('num_epochs', None, 'Number of epochs for training')
flags.DEFINE_string('mode', 'train',
                    'Mode to run: train or eval (default: train)')
flags.DEFINE_string('model_name', 'efficientdet-d1', 'Model name.')
flags.DEFINE_bool('eval_after_train', False, 'Run one eval after the '
                  'training finishes.')
flags.DEFINE_bool('profile', False, 'Profile training performance.')
flags.DEFINE_integer(
    'tf_random_seed', None, 'Sets the TF graph seed for deterministic execution'
    ' across runs (for debugging).')

# For Eval mode
flags.DEFINE_integer('min_eval_interval', 180,
                     'Minimum seconds between evaluations.')
flags.DEFINE_integer(
    'eval_timeout', None,
    'Maximum seconds between checkpoints before evaluation terminates.')

# for train_and_eval mode
flags.DEFINE_bool(
    'run_epoch_in_child_process', False,
    'This option helps to rectify CPU memory leak. If True, every epoch is '
    'run in a separate process for train and eval and memory will be cleared.'
    'Drawback: need to kill 2 processes if trainining needs to be interrupted.')

FLAGS = flags.FLAGS


def main(_):
  if FLAGS.strategy == 'tpu':
    tf.disable_eager_execution()
    tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)
    tpu_grpc_url = tpu_cluster_resolver.get_master()
    tf.Session.reset(tpu_grpc_url)
  else:
    tpu_cluster_resolver = None

  # Check data path
  if FLAGS.mode in ('train', 'train_and_eval'):
    if FLAGS.train_file_pattern is None:
      raise RuntimeError('Must specify --train_file_pattern for train.')
  if FLAGS.mode in ('eval', 'train_and_eval'):
    if FLAGS.val_file_pattern is None:
      raise RuntimeError('Must specify --val_file_pattern for eval.')

  # Parse and override hparams
  config = hparams_config.get_detection_config(FLAGS.model_name)
  config.override(FLAGS.hparams)
  if FLAGS.num_epochs:  # NOTE: remove this flag after updating all docs.
    config.num_epochs = FLAGS.num_epochs

  # Parse image size in case it is in string format.
  config.image_size = utils.parse_image_size(config.image_size)

  # The following is for spatial partitioning. `features` has one tensor while
  # `labels` had 4 + (`max_level` - `min_level` + 1) * 2 tensors. The input
  # partition is performed on `features` and all partitionable tensors of
  # `labels`, see the partition logic below.
  # In the TPUEstimator context, the meaning of `shard` and `replica` is the
  # same; follwing the API, here has mixed use of both.
  if FLAGS.use_spatial_partition:
    # Checks input_partition_dims agrees with num_cores_per_replica.
    if FLAGS.num_cores_per_replica != np.prod(FLAGS.input_partition_dims):
      raise RuntimeError('--num_cores_per_replica must be a product of array'
                         'elements in --input_partition_dims.')

    labels_partition_dims = {
        'mean_num_positives': None,
        'source_ids': None,
        'groundtruth_data': None,
        'image_scales': None,
        'image_masks': None,
    }
    # The Input Partition Logic: We partition only the partition-able tensors.
    feat_sizes = utils.get_feat_sizes(
        config.get('image_size'), config.get('max_level'))
    for level in range(config.get('min_level'), config.get('max_level') + 1):

      def _can_partition(spatial_dim):
        partitionable_index = np.where(
            spatial_dim % np.array(FLAGS.input_partition_dims) == 0)
        return len(partitionable_index[0]) == len(FLAGS.input_partition_dims)

      spatial_dim = feat_sizes[level]
      if _can_partition(spatial_dim['height']) and _can_partition(
          spatial_dim['width']):
        labels_partition_dims['box_targets_%d' %
                              level] = FLAGS.input_partition_dims
        labels_partition_dims['cls_targets_%d' %
                              level] = FLAGS.input_partition_dims
      else:
        labels_partition_dims['box_targets_%d' % level] = None
        labels_partition_dims['cls_targets_%d' % level] = None
    num_cores_per_replica = FLAGS.num_cores_per_replica
    input_partition_dims = [FLAGS.input_partition_dims, labels_partition_dims]
    num_shards = FLAGS.num_cores // num_cores_per_replica
  else:
    num_cores_per_replica = None
    input_partition_dims = None
    num_shards = FLAGS.num_cores

  params = dict(
      config.as_dict(),
      model_name=FLAGS.model_name,
      iterations_per_loop=FLAGS.iterations_per_loop,
      model_dir=FLAGS.model_dir,
      num_shards=num_shards,
      num_examples_per_epoch=FLAGS.num_examples_per_epoch,
      strategy=FLAGS.strategy,
      backbone_ckpt=FLAGS.backbone_ckpt,
      ckpt=FLAGS.ckpt,
      val_json_file=FLAGS.val_json_file,
      testdev_dir=FLAGS.testdev_dir,
      profile=FLAGS.profile,
      mode=FLAGS.mode)
  config_proto = tf.ConfigProto(
      allow_soft_placement=True, log_device_placement=False)
  if FLAGS.strategy != 'tpu':
    if FLAGS.use_xla:
      config_proto.graph_options.optimizer_options.global_jit_level = (
          tf.OptimizerOptions.ON_1)
      config_proto.gpu_options.allow_growth = True

  model_dir = FLAGS.model_dir
  model_fn_instance = det_model_fn.get_model_fn(FLAGS.model_name)
  max_instances_per_image = config.max_instances_per_image
  if FLAGS.eval_samples:
    eval_steps = int((FLAGS.eval_samples + FLAGS.eval_batch_size - 1) //
                     FLAGS.eval_batch_size)
  else:
    eval_steps = None
  total_examples = int(config.num_epochs * FLAGS.num_examples_per_epoch)
  train_steps = total_examples // FLAGS.train_batch_size
  logging.info(params)

  if not tf.io.gfile.exists(model_dir):
    tf.io.gfile.makedirs(model_dir)

  config_file = os.path.join(model_dir, 'config.yaml')
  if not tf.io.gfile.exists(config_file):
    tf.io.gfile.GFile(config_file, 'w').write(str(config))

  train_input_fn = dataloader.InputReader(
      FLAGS.train_file_pattern,
      is_training=True,
      use_fake_data=FLAGS.use_fake_data,
      max_instances_per_image=max_instances_per_image)
  eval_input_fn = dataloader.InputReader(
      FLAGS.val_file_pattern,
      is_training=False,
      use_fake_data=FLAGS.use_fake_data,
      max_instances_per_image=max_instances_per_image)

  if FLAGS.strategy == 'tpu':
    tpu_config = tf.estimator.tpu.TPUConfig(
        FLAGS.iterations_per_loop if FLAGS.strategy == 'tpu' else 1,
        num_cores_per_replica=num_cores_per_replica,
        input_partition_dims=input_partition_dims,
        per_host_input_for_training=tf.estimator.tpu.InputPipelineConfig
        .PER_HOST_V2)
    run_config = tf.estimator.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        model_dir=model_dir,
        log_step_count_steps=FLAGS.iterations_per_loop,
        session_config=config_proto,
        tpu_config=tpu_config,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tf_random_seed=FLAGS.tf_random_seed,
    )
    # TPUEstimator can do both train and eval.
    train_est = tf.estimator.tpu.TPUEstimator(
        model_fn=model_fn_instance,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        config=run_config,
        params=params)
    eval_est = train_est
  else:
    strategy = None
    if FLAGS.strategy == 'gpus':
      strategy = tf.distribute.MirroredStrategy()
    run_config = tf.estimator.RunConfig(
        model_dir=model_dir,
        train_distribute=strategy,
        log_step_count_steps=FLAGS.iterations_per_loop,
        session_config=config_proto,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tf_random_seed=FLAGS.tf_random_seed,
    )

    def get_estimator(global_batch_size):
      params['num_shards'] = getattr(strategy, 'num_replicas_in_sync', 1)
      params['batch_size'] = global_batch_size // params['num_shards']
      return tf.estimator.Estimator(
          model_fn=model_fn_instance, config=run_config, params=params)

    # train and eval need different estimator due to different batch size.
    train_est = get_estimator(FLAGS.train_batch_size)
    eval_est = get_estimator(FLAGS.eval_batch_size)

  # start train/eval flow.
  if FLAGS.mode == 'train':
    train_est.train(input_fn=train_input_fn, max_steps=train_steps)
    if FLAGS.eval_after_train:
      eval_est.evaluate(input_fn=eval_input_fn, steps=eval_steps)

  elif FLAGS.mode == 'eval':
    # Run evaluation when there's a new checkpoint
    for ckpt in tf.train.checkpoints_iterator(
        FLAGS.model_dir,
        min_interval_secs=FLAGS.min_eval_interval,
        timeout=FLAGS.eval_timeout):

      logging.info('Starting to evaluate.')
      try:
        eval_results = eval_est.evaluate(eval_input_fn, steps=eval_steps)
        # Terminate eval job when final checkpoint is reached.
        try:
          current_step = int(os.path.basename(ckpt).split('-')[1])
        except IndexError:
          logging.info('%s has no global step info: stop!', ckpt)
          break

        utils.archive_ckpt(eval_results, eval_results['AP'], ckpt)
        if current_step >= train_steps:
          logging.info('Eval finished step %d/%d', current_step, train_steps)
          break

      except tf.errors.NotFoundError:
        # Checkpoint might be not already deleted by the time eval finished.
        # We simply skip ssuch case.
        logging.info('Checkpoint %s no longer exists, skipping.', ckpt)

  elif FLAGS.mode == 'train_and_eval':
    ckpt = tf.train.latest_checkpoint(FLAGS.model_dir)
    try:
      step = int(os.path.basename(ckpt).split('-')[1])
      current_epoch = (
          step * FLAGS.train_batch_size // FLAGS.num_examples_per_epoch)
      logging.info('found ckpt at step %d (epoch %d)', step, current_epoch)
    except (IndexError, TypeError):
      logging.info('Folder %s has no ckpt with valid step.', FLAGS.model_dir)
      current_epoch = 0

    def run_train_and_eval(e):
      print('\n   =====> Starting training, epoch: %d.' % e)
      train_est.train(
          input_fn=train_input_fn,
          max_steps=e * FLAGS.num_examples_per_epoch // FLAGS.train_batch_size)
      print('\n   =====> Starting evaluation, epoch: %d.' % e)
      eval_results = eval_est.evaluate(input_fn=eval_input_fn, steps=eval_steps)
      ckpt = tf.train.latest_checkpoint(FLAGS.model_dir)
      utils.archive_ckpt(eval_results, eval_results['AP'], ckpt)

    epochs_per_cycle = 1  # higher number has less graph construction overhead.
    for e in range(current_epoch + 1, config.num_epochs + 1, epochs_per_cycle):
      if FLAGS.run_epoch_in_child_process:
        p = multiprocessing.Process(target=run_train_and_eval, args=(e,))
        p.start()
        p.join()
        if p.exitcode != 0:
          return p.exitcode
      else:
        tf.compat.v1.reset_default_graph()
        run_train_and_eval(e)

  else:
    logging.info('Invalid mode: %s', FLAGS.mode)


if __name__ == '__main__':
  app.run(main)
