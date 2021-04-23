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
"""Model function definition, including both architecture and loss."""
import functools
import re
from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow_examples.lite.model_maker.third_party.efficientdet import coco_metric
from tensorflow_examples.lite.model_maker.third_party.efficientdet import efficientdet_arch
from tensorflow_examples.lite.model_maker.third_party.efficientdet import hparams_config
from tensorflow_examples.lite.model_maker.third_party.efficientdet import nms_np
from tensorflow_examples.lite.model_maker.third_party.efficientdet import utils
from tensorflow_examples.lite.model_maker.third_party.efficientdet.keras import anchors
from tensorflow_examples.lite.model_maker.third_party.efficientdet.keras import efficientdet_keras
from tensorflow_examples.lite.model_maker.third_party.efficientdet.keras import postprocess

_DEFAULT_BATCH_SIZE = 64


def update_learning_rate_schedule_parameters(params):
  """Updates params that are related to the learning rate schedule."""
  # params['batch_size'] is per-shard within model_fn if strategy=tpu.
  batch_size = (
      params['batch_size'] * params['num_shards']
      if params['strategy'] in ['tpu', 'gpus'] else params['batch_size'])
  # Learning rate is proportional to the batch size
  params['adjusted_learning_rate'] = (
      params['learning_rate'] * batch_size / _DEFAULT_BATCH_SIZE)

  if 'lr_warmup_init' in params:
    params['adjusted_lr_warmup_init'] = params['lr_warmup_init']

  steps_per_epoch = params['num_examples_per_epoch'] / batch_size
  params['lr_warmup_step'] = int(params['lr_warmup_epoch'] * steps_per_epoch)
  params['first_lr_drop_step'] = int(params['first_lr_drop_epoch'] *
                                     steps_per_epoch)
  params['second_lr_drop_step'] = int(params['second_lr_drop_epoch'] *
                                      steps_per_epoch)
  params['total_steps'] = int(params['num_epochs'] * steps_per_epoch)
  params['steps_per_epoch'] = steps_per_epoch


def stepwise_lr_schedule(adjusted_learning_rate, adjusted_lr_warmup_init,
                         lr_warmup_step, first_lr_drop_step,
                         second_lr_drop_step, global_step):
  """Handles linear scaling rule, gradual warmup, and LR decay."""
  # adjusted_lr_warmup_init is the starting learning rate; LR is linearly
  # scaled up to the full learning rate after `lr_warmup_step` before decaying.
  logging.info('LR schedule method: stepwise')
  linear_warmup = (
      adjusted_lr_warmup_init +
      (tf.cast(global_step, dtype=tf.float32) / lr_warmup_step *
       (adjusted_learning_rate - adjusted_lr_warmup_init)))
  learning_rate = tf.where(global_step < lr_warmup_step, linear_warmup,
                           adjusted_learning_rate)
  lr_schedule = [[1.0, lr_warmup_step], [0.1, first_lr_drop_step],
                 [0.01, second_lr_drop_step]]
  for mult, start_global_step in lr_schedule:
    learning_rate = tf.where(global_step < start_global_step, learning_rate,
                             adjusted_learning_rate * mult)
  return learning_rate


def cosine_lr_schedule(adjusted_lr, adjusted_lr_warmup_init, lr_warmup_step,
                       total_steps, step):
  """Cosine learning rate scahedule."""
  logging.info('LR schedule method: cosine')
  linear_warmup = (
      adjusted_lr_warmup_init +
      (tf.cast(step, dtype=tf.float32) / lr_warmup_step *
       (adjusted_lr - adjusted_lr_warmup_init)))
  decay_steps = tf.cast(total_steps - lr_warmup_step, tf.float32)
  cosine_lr = 0.5 * adjusted_lr * (
      1 + tf.cos(np.pi * tf.cast(step, tf.float32) / decay_steps))
  return tf.where(step < lr_warmup_step, linear_warmup, cosine_lr)


def polynomial_lr_schedule(adjusted_lr, adjusted_lr_warmup_init, lr_warmup_step,
                           power, total_steps, step):
  logging.info('LR schedule method: polynomial')
  linear_warmup = (
      adjusted_lr_warmup_init +
      (tf.cast(step, dtype=tf.float32) / lr_warmup_step *
       (adjusted_lr - adjusted_lr_warmup_init)))
  polynomial_lr = adjusted_lr * tf.pow(
      1 - (tf.cast(step, tf.float32) / total_steps), power)
  return tf.where(step < lr_warmup_step, linear_warmup, polynomial_lr)


def learning_rate_schedule(params, global_step):
  """Learning rate schedule based on global step."""
  lr_decay_method = params['lr_decay_method']
  if lr_decay_method == 'stepwise':
    return stepwise_lr_schedule(params['adjusted_learning_rate'],
                                params['adjusted_lr_warmup_init'],
                                params['lr_warmup_step'],
                                params['first_lr_drop_step'],
                                params['second_lr_drop_step'], global_step)

  if lr_decay_method == 'cosine':
    return cosine_lr_schedule(params['adjusted_learning_rate'],
                              params['adjusted_lr_warmup_init'],
                              params['lr_warmup_step'], params['total_steps'],
                              global_step)

  if lr_decay_method == 'polynomial':
    return polynomial_lr_schedule(params['adjusted_learning_rate'],
                                  params['adjusted_lr_warmup_init'],
                                  params['lr_warmup_step'],
                                  params['poly_lr_power'],
                                  params['total_steps'], global_step)

  if lr_decay_method == 'constant':
    return params['adjusted_learning_rate']

  raise ValueError('unknown lr_decay_method: {}'.format(lr_decay_method))


def focal_loss(y_pred, y_true, alpha, gamma, normalizer, label_smoothing=0.0):
  """Compute the focal loss between `logits` and the golden `target` values.

  Focal loss = -(1-pt)^gamma * log(pt)
  where pt is the probability of being classified to the true class.

  Args:
    y_pred: A float tensor of size [batch, height_in, width_in,
      num_predictions].
    y_true: A float tensor of size [batch, height_in, width_in,
      num_predictions].
    alpha: A float scalar multiplying alpha to the loss from positive examples
      and (1-alpha) to the loss from negative examples.
    gamma: A float scalar modulating loss from hard and easy examples.
    normalizer: Divide loss by this value.
    label_smoothing: Float in [0, 1]. If > `0` then smooth the labels.

  Returns:
    loss: A float32 scalar representing normalized total loss.
  """
  with tf.name_scope('focal_loss'):
    normalizer = tf.cast(normalizer, dtype=y_pred.dtype)

    # compute focal loss multipliers before label smoothing, such that it will
    # not blow up the loss.
    pred_prob = tf.sigmoid(y_pred)
    p_t = (y_true * pred_prob) + ((1 - y_true) * (1 - pred_prob))
    alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
    modulating_factor = (1.0 - p_t)**gamma

    # apply label smoothing for cross_entropy for each entry.
    if label_smoothing:
      y_true = y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing
    ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)

    # compute the final loss and return
    return (1 / normalizer) * alpha_factor * modulating_factor * ce


def _box_loss(box_outputs, box_targets, num_positives, delta=0.1):
  """Computes box regression loss."""
  # delta is typically around the mean value of regression target.
  # for instances, the regression targets of 512x512 input with 6 anchors on
  # P3-P7 pyramid is about [0.1, 0.1, 0.2, 0.2].
  normalizer = num_positives * 4.0
  mask = tf.not_equal(box_targets, 0.0)
  box_loss = tf.losses.huber_loss(
      box_targets,
      box_outputs,
      weights=mask,
      delta=delta,
      reduction=tf.losses.Reduction.SUM)
  box_loss /= normalizer
  return box_loss


def detection_loss(cls_outputs, box_outputs, labels, params):
  """Computes total detection loss.

  Computes total detection loss including box and class loss from all levels.
  Args:
    cls_outputs: an OrderDict with keys representing levels and values
      representing logits in [batch_size, height, width, num_anchors].
    box_outputs: an OrderDict with keys representing levels and values
      representing box regression targets in [batch_size, height, width,
      num_anchors * 4].
    labels: the dictionary that returned from dataloader that includes
      groundtruth targets.
    params: the dictionary including training parameters specified in
      default_haprams function in this file.

  Returns:
    total_loss: an integer tensor representing total loss reducing from
      class and box losses from all levels.
    cls_loss: an integer tensor representing total class loss.
    box_loss: an integer tensor representing total box regression loss.
  """
  # Sum all positives in a batch for normalization and avoid zero
  # num_positives_sum, which would lead to inf loss during training
  num_positives_sum = tf.reduce_sum(labels['mean_num_positives']) + 1.0
  positives_momentum = params.get('positives_momentum', None) or 0
  if positives_momentum > 0:
    # normalize the num_positive_examples for training stability.
    moving_normalizer_var = tf.Variable(
        0.0,
        name='moving_normalizer',
        dtype=tf.float32,
        synchronization=tf.VariableSynchronization.ON_READ,
        trainable=False,
        aggregation=tf.VariableAggregation.MEAN)
    num_positives_sum = tf.keras.backend.moving_average_update(
        moving_normalizer_var,
        num_positives_sum,
        momentum=params['positives_momentum'])
  elif positives_momentum < 0:
    num_positives_sum = utils.cross_replica_mean(num_positives_sum)

  levels = cls_outputs.keys()
  cls_losses = []
  box_losses = []
  for level in levels:
    # Onehot encoding for classification labels.
    cls_targets_at_level = tf.one_hot(
        labels['cls_targets_%d' % level],
        params['num_classes'],
        dtype=cls_outputs[level].dtype)

    if params['data_format'] == 'channels_first':
      bs, _, width, height, _ = cls_targets_at_level.get_shape().as_list()
      cls_targets_at_level = tf.reshape(cls_targets_at_level,
                                        [bs, -1, width, height])
    else:
      bs, width, height, _, _ = cls_targets_at_level.get_shape().as_list()
      cls_targets_at_level = tf.reshape(cls_targets_at_level,
                                        [bs, width, height, -1])
    box_targets_at_level = labels['box_targets_%d' % level]

    cls_loss = focal_loss(
        cls_outputs[level],
        cls_targets_at_level,
        params['alpha'],
        params['gamma'],
        normalizer=num_positives_sum,
        label_smoothing=params['label_smoothing'])

    if params['data_format'] == 'channels_first':
      cls_loss = tf.reshape(cls_loss,
                            [bs, -1, width, height, params['num_classes']])
    else:
      cls_loss = tf.reshape(cls_loss,
                            [bs, width, height, -1, params['num_classes']])

    cls_loss *= tf.cast(
        tf.expand_dims(tf.not_equal(labels['cls_targets_%d' % level], -2), -1),
        cls_loss.dtype)
    cls_loss_sum = tf.reduce_sum(cls_loss)
    cls_losses.append(tf.cast(cls_loss_sum, tf.float32))

    if params['box_loss_weight']:
      box_losses.append(
          _box_loss(
              box_outputs[level],
              box_targets_at_level,
              num_positives_sum,
              delta=params['delta']))

  # Sum per level losses to total loss.
  cls_loss = tf.add_n(cls_losses)
  box_loss = tf.add_n(box_losses) if box_losses else tf.constant(0.)

  total_loss = (
      cls_loss + params['box_loss_weight'] * box_loss)

  return total_loss, cls_loss, box_loss


def reg_l2_loss(weight_decay, regex=r'.*(kernel|weight):0$'):
  """Return regularization l2 loss loss."""
  var_match = re.compile(regex)
  return weight_decay * tf.add_n([
      tf.nn.l2_loss(v)
      for v in tf.trainable_variables()
      if var_match.match(v.name)
  ])


@tf.autograph.experimental.do_not_convert
def _model_fn(features, labels, mode, params, model, variable_filter_fn=None):
  """Model definition entry.

  Args:
    features: the input image tensor with shape [batch_size, height, width, 3].
      The height and width are fixed and equal.
    labels: the input labels in a dictionary. The labels include class targets
      and box targets which are dense label maps. The labels are generated from
      get_input_fn function in data/dataloader.py
    mode: the mode of TPUEstimator including TRAIN and EVAL.
    params: the dictionary defines hyperparameters of model. The default
      settings are in default_hparams function in this file.
    model: the model outputs class logits and box regression outputs.
    variable_filter_fn: the filter function that takes trainable_variables and
      returns the variable list after applying the filter rule.

  Returns:
    tpu_spec: the TPUEstimatorSpec to run training, evaluation, or prediction.

  Raises:
    RuntimeError: if both ckpt and backbone_ckpt are set.
  """
  is_tpu = params['strategy'] == 'tpu'
  if params['img_summary_steps']:
    utils.image('input_image', features, is_tpu)
  training_hooks = []
  params['is_training_bn'] = (mode == tf.estimator.ModeKeys.TRAIN)

  if params['use_keras_model']:

    def model_fn(inputs):
      model = efficientdet_keras.EfficientDetNet(
          config=hparams_config.Config(params))
      cls_out_list, box_out_list = model(inputs, params['is_training_bn'])
      cls_outputs, box_outputs = {}, {}
      for i in range(params['min_level'], params['max_level'] + 1):
        cls_outputs[i] = cls_out_list[i - params['min_level']]
        box_outputs[i] = box_out_list[i - params['min_level']]
      return cls_outputs, box_outputs
  else:
    model_fn = functools.partial(model, config=hparams_config.Config(params))

  precision = utils.get_precision(params['strategy'], params['mixed_precision'])
  cls_outputs, box_outputs = utils.build_model_with_precision(
      precision, model_fn, features)

  levels = cls_outputs.keys()
  for level in levels:
    cls_outputs[level] = tf.cast(cls_outputs[level], tf.float32)
    box_outputs[level] = tf.cast(box_outputs[level], tf.float32)

  # Set up training loss and learning rate.
  update_learning_rate_schedule_parameters(params)
  global_step = tf.train.get_or_create_global_step()
  learning_rate = learning_rate_schedule(params, global_step)

  # cls_loss and box_loss are for logging. only total_loss is optimized.
  det_loss, cls_loss, box_loss = detection_loss(
      cls_outputs, box_outputs, labels, params)
  reg_l2loss = reg_l2_loss(params['weight_decay'])
  total_loss = det_loss + reg_l2loss

  if mode == tf.estimator.ModeKeys.TRAIN:
    utils.scalar('lrn_rate', learning_rate, is_tpu)
    utils.scalar('trainloss/cls_loss', cls_loss, is_tpu)
    utils.scalar('trainloss/box_loss', box_loss, is_tpu)
    utils.scalar('trainloss/det_loss', det_loss, is_tpu)
    utils.scalar('trainloss/reg_l2_loss', reg_l2loss, is_tpu)
    utils.scalar('trainloss/loss', total_loss, is_tpu)
    train_epochs = tf.cast(global_step, tf.float32) / params['steps_per_epoch']
    utils.scalar('train_epochs', train_epochs, is_tpu)

  moving_average_decay = params['moving_average_decay']
  if moving_average_decay:
    ema = tf.train.ExponentialMovingAverage(
        decay=moving_average_decay, num_updates=global_step)
    ema_vars = utils.get_ema_vars()

  if mode == tf.estimator.ModeKeys.TRAIN:
    if params['optimizer'].lower() == 'sgd':
      optimizer = tf.train.MomentumOptimizer(
          learning_rate, momentum=params['momentum'])
    elif params['optimizer'].lower() == 'adam':
      optimizer = tf.train.AdamOptimizer(learning_rate)
    else:
      raise ValueError('optimizers should be adam or sgd')

    if is_tpu:
      optimizer = tf.tpu.CrossShardOptimizer(optimizer)

    # Batch norm requires update_ops to be added as a train_op dependency.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    var_list = tf.trainable_variables()
    if variable_filter_fn:
      var_list = variable_filter_fn(var_list)

    if params.get('clip_gradients_norm', None):
      logging.info('clip gradients norm by %f', params['clip_gradients_norm'])
      grads_and_vars = optimizer.compute_gradients(total_loss, var_list)
      with tf.name_scope('clip'):
        grads = [gv[0] for gv in grads_and_vars]
        tvars = [gv[1] for gv in grads_and_vars]
        # First clip each variable's norm, then clip global norm.
        clip_norm = abs(params['clip_gradients_norm'])
        clipped_grads = [
            tf.clip_by_norm(g, clip_norm) if g is not None else None
            for g in grads
        ]
        clipped_grads, _ = tf.clip_by_global_norm(clipped_grads, clip_norm)
        utils.scalar('gradient_norm', tf.linalg.global_norm(clipped_grads),
                     is_tpu)
        grads_and_vars = list(zip(clipped_grads, tvars))

      with tf.control_dependencies(update_ops):
        train_op = optimizer.apply_gradients(grads_and_vars, global_step)
    else:
      with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(
            total_loss, global_step, var_list=var_list)

    if moving_average_decay:
      with tf.control_dependencies([train_op]):
        train_op = ema.apply(ema_vars)

  else:
    train_op = None

  eval_metrics = None
  if mode == tf.estimator.ModeKeys.EVAL:

    def metric_fn(**kwargs):
      """Returns a dictionary that has the evaluation metrics."""
      if params['nms_configs'].get('pyfunc', True):
        detections_bs = []
        nms_configs = params['nms_configs']
        for index in range(kwargs['boxes'].shape[0]):
          detections = tf.numpy_function(
              functools.partial(nms_np.per_class_nms, nms_configs=nms_configs),
              [
                  kwargs['boxes'][index],
                  kwargs['scores'][index],
                  kwargs['classes'][index],
                  tf.slice(kwargs['image_ids'], [index], [1]),
                  tf.slice(kwargs['image_scales'], [index], [1]),
                  params['num_classes'],
                  nms_configs['max_output_size'],
              ], tf.float32)
          detections_bs.append(detections)
        detections_bs = postprocess.transform_detections(
            tf.stack(detections_bs))
      else:
        # These two branches should be equivalent, but currently they are not.
        # TODO(tanmingxing): enable the non_pyfun path after bug fix.
        nms_boxes, nms_scores, nms_classes, _ = postprocess.per_class_nms(
            params, kwargs['boxes'], kwargs['scores'], kwargs['classes'],
            kwargs['image_scales'])
        img_ids = tf.cast(
            tf.expand_dims(kwargs['image_ids'], -1), nms_scores.dtype)
        detections_bs = [
            img_ids * tf.ones_like(nms_scores),
            nms_boxes[:, :, 1],
            nms_boxes[:, :, 0],
            nms_boxes[:, :, 3] - nms_boxes[:, :, 1],
            nms_boxes[:, :, 2] - nms_boxes[:, :, 0],
            nms_scores,
            nms_classes,
        ]
        detections_bs = tf.stack(detections_bs, axis=-1, name='detnections')

      if params.get('testdev_dir', None):
        logging.info('Eval testdev_dir %s', params['testdev_dir'])
        eval_metric = coco_metric.EvaluationMetric(
            testdev_dir=params['testdev_dir'])
        coco_metrics = eval_metric.estimator_metric_fn(detections_bs,
                                                       tf.zeros([1]))
      else:
        logging.info('Eval val with groudtruths %s.', params['val_json_file'])
        eval_metric = coco_metric.EvaluationMetric(
            filename=params['val_json_file'], label_map=params['label_map'])
        coco_metrics = eval_metric.estimator_metric_fn(
            detections_bs, kwargs['groundtruth_data'])

      # Add metrics to output.
      cls_loss = tf.metrics.mean(kwargs['cls_loss_repeat'])
      box_loss = tf.metrics.mean(kwargs['box_loss_repeat'])
      output_metrics = {
          'cls_loss': cls_loss,
          'box_loss': box_loss,
      }
      output_metrics.update(coco_metrics)
      return output_metrics

    cls_loss_repeat = tf.reshape(
        tf.tile(tf.expand_dims(cls_loss, 0), [
            params['batch_size'],
        ]), [params['batch_size'], 1])
    box_loss_repeat = tf.reshape(
        tf.tile(tf.expand_dims(box_loss, 0), [
            params['batch_size'],
        ]), [params['batch_size'], 1])

    cls_outputs = postprocess.to_list(cls_outputs)
    box_outputs = postprocess.to_list(box_outputs)
    params['nms_configs']['max_nms_inputs'] = anchors.MAX_DETECTION_POINTS
    boxes, scores, classes = postprocess.pre_nms(params, cls_outputs,
                                                 box_outputs)
    metric_fn_inputs = {
        'cls_loss_repeat': cls_loss_repeat,
        'box_loss_repeat': box_loss_repeat,
        'image_ids': labels['source_ids'],
        'groundtruth_data': labels['groundtruth_data'],
        'image_scales': labels['image_scales'],
        'boxes': boxes,
        'scores': scores,
        'classes': classes,
    }
    eval_metrics = (metric_fn, metric_fn_inputs)

  checkpoint = params.get('ckpt') or params.get('backbone_ckpt')

  if checkpoint and mode == tf.estimator.ModeKeys.TRAIN:
    # Initialize the model from an EfficientDet or backbone checkpoint.
    if params.get('ckpt') and params.get('backbone_ckpt'):
      raise RuntimeError(
          '--backbone_ckpt and --checkpoint are mutually exclusive')

    if params.get('backbone_ckpt'):
      var_scope = params['backbone_name'] + '/'
      if params['ckpt_var_scope'] is None:
        # Use backbone name as default checkpoint scope.
        ckpt_scope = params['backbone_name'] + '/'
      else:
        ckpt_scope = params['ckpt_var_scope'] + '/'
    else:
      # Load every var in the given checkpoint
      var_scope = ckpt_scope = '/'

    def scaffold_fn():
      """Loads pretrained model through scaffold function."""
      logging.info('restore variables from %s', checkpoint)

      var_map = utils.get_ckpt_var_map(
          ckpt_path=checkpoint,
          ckpt_scope=ckpt_scope,
          var_scope=var_scope,
          skip_mismatch=params['skip_mismatch'])

      tf.train.init_from_checkpoint(checkpoint, var_map)
      return tf.train.Scaffold()
  elif mode == tf.estimator.ModeKeys.EVAL and moving_average_decay:

    def scaffold_fn():
      """Load moving average variables for eval."""
      logging.info('Load EMA vars with ema_decay=%f', moving_average_decay)
      restore_vars_dict = ema.variables_to_restore(ema_vars)
      saver = tf.train.Saver(restore_vars_dict)
      return tf.train.Scaffold(saver=saver)
  else:
    scaffold_fn = None

  if is_tpu:
    return tf.estimator.tpu.TPUEstimatorSpec(
        mode=mode,
        loss=total_loss,
        train_op=train_op,
        eval_metrics=eval_metrics,
        host_call=utils.get_tpu_host_call(global_step, params),
        scaffold_fn=scaffold_fn,
        training_hooks=training_hooks)
  else:
    # Profile every 1K steps.
    if params.get('profile', False):
      profile_hook = tf.estimator.ProfilerHook(
          save_steps=1000, output_dir=params['model_dir'], show_memory=True)
      training_hooks.append(profile_hook)

      # Report memory allocation if OOM; it will slow down the running.
      class OomReportingHook(tf.estimator.SessionRunHook):

        def before_run(self, run_context):
          return tf.estimator.SessionRunArgs(
              fetches=[],
              options=tf.RunOptions(report_tensor_allocations_upon_oom=True))

      training_hooks.append(OomReportingHook())

    logging_hook = tf.estimator.LoggingTensorHook(
        {
            'step': global_step,
            'det_loss': det_loss,
            'cls_loss': cls_loss,
            'box_loss': box_loss,
        },
        every_n_iter=params.get('iterations_per_loop', 100),
    )
    training_hooks.append(logging_hook)

    eval_metric_ops = (
        eval_metrics[0](**eval_metrics[1]) if eval_metrics else None)
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=total_loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops,
        scaffold=scaffold_fn() if scaffold_fn else None,
        training_hooks=training_hooks)


def efficientdet_model_fn(features, labels, mode, params):
  """EfficientDet model."""
  variable_filter_fn = functools.partial(
      efficientdet_arch.freeze_vars, pattern=params['var_freeze_expr'])
  return _model_fn(
      features,
      labels,
      mode,
      params,
      model=efficientdet_arch.efficientdet,
      variable_filter_fn=variable_filter_fn)


def get_model_arch(model_name='efficientdet-d0'):
  """Get model architecture for a given model name."""
  if 'efficientdet' in model_name:
    return efficientdet_arch.efficientdet

  raise ValueError('Invalide model name {}'.format(model_name))


def get_model_fn(model_name='efficientdet-d0'):
  """Get model fn for a given model name."""
  if 'efficientdet' in model_name:
    return efficientdet_model_fn

  raise ValueError('Invalide model name {}'.format(model_name))
