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
"""Training related libraries."""
import math
import os
import re
from absl import logging
import neural_structured_learning as nsl
import numpy as np

import tensorflow as tf
from tensorflow_addons.callbacks import AverageModelCheckpoint
import tensorflow_hub as hub

from tensorflow_examples.lite.model_maker.third_party.efficientdet import coco_metric
from tensorflow_examples.lite.model_maker.third_party.efficientdet import inference
from tensorflow_examples.lite.model_maker.third_party.efficientdet import iou_utils
from tensorflow_examples.lite.model_maker.third_party.efficientdet import utils
from tensorflow_examples.lite.model_maker.third_party.efficientdet.keras import anchors
from tensorflow_examples.lite.model_maker.third_party.efficientdet.keras import efficientdet_keras
from tensorflow_examples.lite.model_maker.third_party.efficientdet.keras import label_util
from tensorflow_examples.lite.model_maker.third_party.efficientdet.keras import postprocess
from tensorflow_examples.lite.model_maker.third_party.efficientdet.keras import util_keras
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper


def _collect_prunable_layers(model):
  """Recursively collect the prunable layers in the model."""
  prunable_layers = []
  for layer in model._flatten_layers(recursive=False, include_self=False):  # pylint: disable=protected-access
    # A keras model may have other models as layers.
    if isinstance(layer, pruning_wrapper.PruneLowMagnitude):
      prunable_layers.append(layer)
    elif isinstance(layer, (tf.keras.Model, tf.keras.layers.Layer)):
      prunable_layers += _collect_prunable_layers(layer)

  return prunable_layers


class UpdatePruningStep(tf.keras.callbacks.Callback):
  """Keras callback which updates pruning wrappers with the optimizer step.

  This callback must be used when training a model which needs to be pruned. Not
  doing so will throw an error.
  Example:
  ```python
  model.fit(x, y,
      callbacks=[UpdatePruningStep()])
  ```
  """

  def __init__(self):
    super(UpdatePruningStep, self).__init__()
    self.prunable_layers = []

  def on_train_begin(self, logs=None):
    # Collect all the prunable layers in the model.
    self.prunable_layers = _collect_prunable_layers(self.model)
    self.step = tf.keras.backend.get_value(self.model.optimizer.iterations)

  def on_train_batch_begin(self, batch, logs=None):
    tuples = []

    for layer in self.prunable_layers:
      if layer.built:
        tuples.append((layer.pruning_step, self.step))

    tf.keras.backend.batch_set_value(tuples)
    self.step = self.step + 1

  def on_epoch_end(self, batch, logs=None):
    # At the end of every epoch, remask the weights. This ensures that when
    # the model is saved after completion, the weights represent mask*weights.
    weight_mask_ops = []

    for layer in self.prunable_layers:
      if layer.built and isinstance(layer, pruning_wrapper.PruneLowMagnitude):
        if tf.executing_eagerly():
          layer.pruning_obj.weight_mask_op()
        else:
          weight_mask_ops.append(layer.pruning_obj.weight_mask_op())

    tf.keras.backend.batch_get_value(weight_mask_ops)


class PruningSummaries(tf.keras.callbacks.TensorBoard):
  """A Keras callback for adding pruning summaries to tensorboard.

  Logs the sparsity(%) and threshold at a given iteration step.
  """

  def __init__(self, log_dir, update_freq='epoch', **kwargs):
    if not isinstance(log_dir, str) or not log_dir:
      raise ValueError(
          '`log_dir` must be a non-empty string. You passed `log_dir`='
          '{input}.'.format(input=log_dir))

    super().__init__(log_dir=log_dir, update_freq=update_freq, **kwargs)

    log_dir = self.log_dir + '/metrics'
    self._file_writer = tf.summary.create_file_writer(log_dir)

  def _log_pruning_metrics(self, logs, step):
    with self._file_writer.as_default():
      for name, value in logs.items():
        tf.summary.scalar(name, value, step=step)

      self._file_writer.flush()

  def on_epoch_begin(self, epoch, logs=None):
    if logs is not None:
      super().on_epoch_begin(epoch, logs)

    pruning_logs = {}
    params = []
    prunable_layers = _collect_prunable_layers(self.model)
    for layer in prunable_layers:
      for _, mask, threshold in layer.pruning_vars:
        params.append(mask)
        params.append(threshold)

    params.append(self.model.optimizer.iterations)

    values = tf.keras.backend.batch_get_value(params)
    iteration = values[-1]
    del values[-1]
    del params[-1]

    param_value_pairs = list(zip(params, values))

    for mask, mask_value in param_value_pairs[::2]:
      pruning_logs.update({mask.name + '/sparsity': 1 - np.mean(mask_value)})

    for threshold, threshold_value in param_value_pairs[1::2]:
      pruning_logs.update({threshold.name + '/threshold': threshold_value})

    self._log_pruning_metrics(pruning_logs, iteration)


def update_learning_rate_schedule_parameters(params):
  """Updates params that are related to the learning rate schedule."""
  batch_size = params['batch_size']
  # Learning rate is proportional to the batch size
  params['adjusted_learning_rate'] = (params['learning_rate'] * batch_size / 64)
  steps_per_epoch = params['steps_per_epoch']
  params['lr_warmup_step'] = int(params['lr_warmup_epoch'] * steps_per_epoch)
  params['first_lr_drop_step'] = int(params['first_lr_drop_epoch'] *
                                     steps_per_epoch)
  params['second_lr_drop_step'] = int(params['second_lr_drop_epoch'] *
                                      steps_per_epoch)
  params['total_steps'] = int(params['num_epochs'] * steps_per_epoch)


class StepwiseLrSchedule(tf.optimizers.schedules.LearningRateSchedule):
  """Stepwise learning rate schedule."""

  def __init__(self, adjusted_lr: float, lr_warmup_init: float,
               lr_warmup_step: int, first_lr_drop_step: int,
               second_lr_drop_step: int):
    """Build a StepwiseLrSchedule.

    Args:
      adjusted_lr: `float`, The initial learning rate.
      lr_warmup_init: `float`, The warm up learning rate.
      lr_warmup_step: `int`, The warm up step.
      first_lr_drop_step: `int`, First lr decay step.
      second_lr_drop_step: `int`, Second lr decay step.
    """
    super().__init__()
    logging.info('LR schedule method: stepwise')
    self.adjusted_lr = adjusted_lr
    self.lr_warmup_init = lr_warmup_init
    self.lr_warmup_step = lr_warmup_step
    self.first_lr_drop_step = first_lr_drop_step
    self.second_lr_drop_step = second_lr_drop_step

  def __call__(self, step):
    linear_warmup = (
        self.lr_warmup_init +
        (tf.cast(step, dtype=tf.float32) / self.lr_warmup_step *
         (self.adjusted_lr - self.lr_warmup_init)))
    learning_rate = tf.where(step < self.lr_warmup_step, linear_warmup,
                             self.adjusted_lr)
    lr_schedule = [[1.0, self.lr_warmup_step], [0.1, self.first_lr_drop_step],
                   [0.01, self.second_lr_drop_step]]
    for mult, start_global_step in lr_schedule:
      learning_rate = tf.where(step < start_global_step, learning_rate,
                               self.adjusted_lr * mult)
    return learning_rate


class CosineLrSchedule(tf.optimizers.schedules.LearningRateSchedule):
  """Cosine learning rate schedule."""

  def __init__(self, adjusted_lr: float, lr_warmup_init: float,
               lr_warmup_step: int, total_steps: int):
    """Build a CosineLrSchedule.

    Args:
      adjusted_lr: `float`, The initial learning rate.
      lr_warmup_init: `float`, The warm up learning rate.
      lr_warmup_step: `int`, The warm up step.
      total_steps: `int`, Total train steps.
    """
    super().__init__()
    logging.info('LR schedule method: cosine')
    self.adjusted_lr = adjusted_lr
    self.lr_warmup_init = lr_warmup_init
    self.lr_warmup_step = lr_warmup_step
    self.decay_steps = tf.cast(total_steps - lr_warmup_step, tf.float32)

  def __call__(self, step):
    linear_warmup = (
        self.lr_warmup_init +
        (tf.cast(step, dtype=tf.float32) / self.lr_warmup_step *
         (self.adjusted_lr - self.lr_warmup_init)))
    cosine_lr = 0.5 * self.adjusted_lr * (
        1 + tf.cos(math.pi * tf.cast(step, tf.float32) / self.decay_steps))
    return tf.where(step < self.lr_warmup_step, linear_warmup, cosine_lr)


class PolynomialLrSchedule(tf.optimizers.schedules.LearningRateSchedule):
  """Polynomial learning rate schedule."""

  def __init__(self, adjusted_lr: float, lr_warmup_init: float,
               lr_warmup_step: int, power: float, total_steps: int):
    """Build a PolynomialLrSchedule.

    Args:
      adjusted_lr: `float`, The initial learning rate.
      lr_warmup_init: `float`, The warm up learning rate.
      lr_warmup_step: `int`, The warm up step.
      power: `float`, power.
      total_steps: `int`, Total train steps.
    """
    super().__init__()
    logging.info('LR schedule method: polynomial')
    self.adjusted_lr = adjusted_lr
    self.lr_warmup_init = lr_warmup_init
    self.lr_warmup_step = lr_warmup_step
    self.power = power
    self.total_steps = total_steps

  def __call__(self, step):
    linear_warmup = (
        self.lr_warmup_init +
        (tf.cast(step, dtype=tf.float32) / self.lr_warmup_step *
         (self.adjusted_lr - self.lr_warmup_init)))
    polynomial_lr = self.adjusted_lr * tf.pow(
        1 - (tf.cast(step, dtype=tf.float32) / self.total_steps), self.power)
    return tf.where(step < self.lr_warmup_step, linear_warmup, polynomial_lr)


def learning_rate_schedule(params):
  """Learning rate schedule based on global step."""
  update_learning_rate_schedule_parameters(params)
  lr_decay_method = params['lr_decay_method']
  if lr_decay_method == 'stepwise':
    return StepwiseLrSchedule(params['adjusted_learning_rate'],
                              params['lr_warmup_init'],
                              params['lr_warmup_step'],
                              params['first_lr_drop_step'],
                              params['second_lr_drop_step'])

  if lr_decay_method == 'cosine':
    return CosineLrSchedule(params['adjusted_learning_rate'],
                            params['lr_warmup_init'], params['lr_warmup_step'],
                            params['total_steps'])

  if lr_decay_method == 'polynomial':
    return PolynomialLrSchedule(params['adjusted_learning_rate'],
                                params['lr_warmup_init'],
                                params['lr_warmup_step'],
                                params['poly_lr_power'], params['total_steps'])

  raise ValueError('unknown lr_decay_method: {}'.format(lr_decay_method))


def get_optimizer(params):
  """Get optimizer."""
  learning_rate = learning_rate_schedule(params)
  momentum = params['momentum']
  if params['optimizer'].lower() == 'sgd':
    logging.info('Use SGD optimizer')
    optimizer = tf.keras.optimizers.SGD(learning_rate, momentum=momentum)
  elif params['optimizer'].lower() == 'adam':
    logging.info('Use Adam optimizer')
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=momentum)
  else:
    raise ValueError('optimizers should be adam or sgd')

  moving_average_decay = params['moving_average_decay']
  if moving_average_decay:
    # TODO(tanmingxing): potentially add dynamic_decay for new tfa release.
    from tensorflow_addons import optimizers as tfa_optimizers  # pylint: disable=g-import-not-at-top
    optimizer = tfa_optimizers.MovingAverage(
        optimizer, average_decay=moving_average_decay, dynamic_decay=True)
  precision = utils.get_precision(params['strategy'], params['mixed_precision'])
  if precision == 'mixed_float16' and params['loss_scale']:
    optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(
        optimizer,
        loss_scale=tf.mixed_precision.experimental.DynamicLossScale(
            params['loss_scale']))
  return optimizer


class COCOCallback(tf.keras.callbacks.Callback):
  """A utility for COCO eval callback."""

  def __init__(self, test_dataset, update_freq=None):
    super().__init__()
    self.test_dataset = test_dataset
    self.update_freq = update_freq

  def set_model(self, model: tf.keras.Model):
    self.model = model
    config = model.config
    self.config = config
    label_map = label_util.get_label_map(config.label_map)
    log_dir = os.path.join(config.model_dir, 'coco')
    self.file_writer = tf.summary.create_file_writer(log_dir)
    self.evaluator = coco_metric.EvaluationMetric(
        filename=config.val_json_file, label_map=label_map)

  @tf.function
  def _get_detections(self, images, labels):
    cls_outputs, box_outputs = self.model(images, training=False)
    detections = postprocess.generate_detections(self.config,
                                                 cls_outputs,
                                                 box_outputs,
                                                 labels['image_scales'],
                                                 labels['source_ids'])
    tf.numpy_function(self.evaluator.update_state,
                      [labels['groundtruth_data'],
                       postprocess.transform_detections(detections)], [])

  def on_epoch_end(self, epoch, logs=None):
    epoch += 1
    if self.update_freq and epoch % self.update_freq == 0:
      self.evaluator.reset_states()
      strategy = tf.distribute.get_strategy()
      count = self.config.eval_samples // self.config.batch_size
      dataset = self.test_dataset.take(count)
      dataset = strategy.experimental_distribute_dataset(dataset)
      for (images, labels) in dataset:
        strategy.run(self._get_detections, (images, labels))
      metrics = self.evaluator.result()
      eval_results = {}
      with self.file_writer.as_default(), tf.summary.record_if(True):
        for i, name in enumerate(self.evaluator.metric_names):
          tf.summary.scalar(name, metrics[i], step=epoch)
          eval_results[name] = metrics[i]
      return eval_results


class DisplayCallback(tf.keras.callbacks.Callback):
  """Display inference result callback."""

  def __init__(self, sample_image, output_dir, update_freq=None):
    super().__init__()
    image_file = tf.io.read_file(sample_image)
    self.sample_image = tf.expand_dims(
        tf.image.decode_jpeg(image_file, channels=3), axis=0)
    self.update_freq = update_freq
    self.output_dir = output_dir

  def set_model(self, model: tf.keras.Model):
    self.model = model
    config = model.config
    log_dir = os.path.join(config.model_dir, 'test_images')
    self.file_writer = tf.summary.create_file_writer(log_dir)
    self.min_score_thresh = config.nms_configs['score_thresh'] or 0.4
    self.max_boxes_to_draw = config.nms_configs['max_output_size'] or 100

  def on_train_batch_end(self, batch, logs=None):
    if self.update_freq and batch % self.update_freq == 0:
      self._draw_inference(batch)

  def _draw_inference(self, step):
    self.model.__class__ = efficientdet_keras.EfficientDetModel
    results = self.model(self.sample_image, training=False)
    boxes, scores, classes, valid_len = tf.nest.map_structure(np.array, results)
    length = valid_len[0]
    image = inference.visualize_image(
        self.sample_image[0],
        boxes[0][:length],
        classes[0].astype(np.int)[:length],
        scores[0][:length],
        label_map=self.model.config.label_map,
        min_score_thresh=self.min_score_thresh,
        max_boxes_to_draw=self.max_boxes_to_draw)

    with self.file_writer.as_default():
      tf.summary.image('Test image', tf.expand_dims(image, axis=0), step=step)
    self.model.__class__ = EfficientDetNetTrain


def get_callbacks(params, val_dataset=None):
  """Get callbacks for given params."""
  if params['moving_average_decay']:
    avg_callback = AverageModelCheckpoint(
        filepath=os.path.join(params['model_dir'], 'emackpt-{epoch:d}'),
        verbose=1,
        save_weights_only=True,
        update_weights=False)
    callbacks = [avg_callback]
  else:
    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(params['model_dir'], 'ckpt-{epoch:d}'),
        verbose=1,
        save_weights_only=True)
    callbacks = [ckpt_callback]
  if params['model_optimizations'] and 'prune' in params['model_optimizations']:
    prune_callback = UpdatePruningStep()
    prune_summaries = PruningSummaries(
        log_dir=params['model_dir'],
        update_freq=params['steps_per_execution'],
        profile_batch=2 if params['profile'] else 0)
    callbacks += [prune_callback, prune_summaries]
  else:
    tb_callback = tf.keras.callbacks.TensorBoard(
        log_dir=params['model_dir'],
        update_freq=params['steps_per_execution'],
        profile_batch=2 if params['profile'] else 0)
    callbacks.append(tb_callback)
  if params.get('sample_image', None):
    display_callback = DisplayCallback(
        params.get('sample_image', None), params['model_dir'],
        params['img_summary_steps'])
    callbacks.append(display_callback)
  if (params.get('map_freq', None) and val_dataset and
      params['strategy'] != 'tpu'):
    coco_callback = COCOCallback(val_dataset, params['map_freq'])
    callbacks.append(coco_callback)
  return callbacks


class AdversarialLoss(tf.keras.losses.Loss):
  """Adversarial keras loss wrapper."""

  # TODO(fsx950223): WIP
  def __init__(self, adv_config, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.adv_config = adv_config
    self.model = None
    self.loss_fn = None
    self.tape = None
    self.built = False

  def build(self, model, loss_fn, tape):
    self.model = model
    self.loss_fn = loss_fn
    self.tape = tape
    self.built = True

  def call(self, features, y, y_pred, labeled_loss):
    return self.adv_config.multiplier * nsl.keras.adversarial_loss(
        features,
        y,
        self.model,
        self.loss_fn,
        predictions=y_pred,
        labeled_loss=self.labeled_loss,
        gradient_tape=self.tape)


class FocalLoss(tf.keras.losses.Loss):
  """Compute the focal loss between `logits` and the golden `target` values.

  Focal loss = -(1-pt)^gamma * log(pt)
  where pt is the probability of being classified to the true class.
  """

  def __init__(self, alpha, gamma, label_smoothing=0.0, **kwargs):
    """Initialize focal loss.

    Args:
      alpha: A float32 scalar multiplying alpha to the loss from positive
        examples and (1-alpha) to the loss from negative examples.
      gamma: A float32 scalar modulating loss from hard and easy examples.
      label_smoothing: Float in [0, 1]. If > `0` then smooth the labels.
      **kwargs: other params.
    """
    super().__init__(**kwargs)
    self.alpha = alpha
    self.gamma = gamma
    self.label_smoothing = label_smoothing

  @tf.autograph.experimental.do_not_convert
  def call(self, y, y_pred):
    """Compute focal loss for y and y_pred.

    Args:
      y: A tuple of (normalizer, y_true), where y_true is the target class.
      y_pred: A float32 tensor [batch, height_in, width_in, num_predictions].

    Returns:
      the focal loss.
    """
    normalizer, y_true = y
    alpha = tf.convert_to_tensor(self.alpha, dtype=y_pred.dtype)
    gamma = tf.convert_to_tensor(self.gamma, dtype=y_pred.dtype)

    # compute focal loss multipliers before label smoothing, such that it will
    # not blow up the loss.
    pred_prob = tf.sigmoid(y_pred)
    p_t = (y_true * pred_prob) + ((1 - y_true) * (1 - pred_prob))
    alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
    modulating_factor = (1.0 - p_t)**gamma

    # apply label smoothing for cross_entropy for each entry.
    y_true = y_true * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
    ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)

    # compute the final loss and return
    return alpha_factor * modulating_factor * ce / normalizer


class BoxLoss(tf.keras.losses.Loss):
  """L2 box regression loss."""

  def __init__(self, delta=0.1, **kwargs):
    """Initialize box loss.

    Args:
      delta: `float`, the point where the huber loss function changes from a
        quadratic to linear. It is typically around the mean value of regression
        target. For instances, the regression targets of 512x512 input with 6
        anchors on P3-P7 pyramid is about [0.1, 0.1, 0.2, 0.2].
      **kwargs: other params.
    """
    super().__init__(**kwargs)
    self.huber = tf.keras.losses.Huber(
        delta, reduction=tf.keras.losses.Reduction.NONE)

  @tf.autograph.experimental.do_not_convert
  def call(self, y_true, box_outputs):
    num_positives, box_targets = y_true
    normalizer = num_positives * 4.0
    mask = tf.cast(box_targets != 0.0, box_outputs.dtype)
    box_targets = tf.expand_dims(box_targets, axis=-1)
    box_outputs = tf.expand_dims(box_outputs, axis=-1)
    # TODO(fsx950223): remove cast when huber loss dtype is fixed.
    box_loss = tf.cast(self.huber(box_targets, box_outputs),
                       box_outputs.dtype) * mask
    box_loss = tf.reduce_sum(box_loss) / normalizer
    return box_loss


class BoxIouLoss(tf.keras.losses.Loss):
  """Box iou loss."""

  def __init__(self, iou_loss_type, min_level, max_level, num_scales,
               aspect_ratios, anchor_scale, image_size, **kwargs):
    super().__init__(**kwargs)
    self.iou_loss_type = iou_loss_type
    self.input_anchors = anchors.Anchors(min_level, max_level, num_scales,
                                         aspect_ratios, anchor_scale,
                                         image_size)

  @tf.autograph.experimental.do_not_convert
  def call(self, y_true, box_outputs):
    anchor_boxes = tf.tile(
        self.input_anchors.boxes,
        [box_outputs.shape[0] // self.input_anchors.boxes.shape[0], 1])
    num_positives, box_targets = y_true
    normalizer = num_positives * 4.0
    mask = tf.cast(box_targets != 0.0, box_outputs.dtype)
    box_outputs = anchors.decode_box_outputs(box_outputs, anchor_boxes) * mask
    box_targets = anchors.decode_box_outputs(box_targets, anchor_boxes) * mask
    box_iou_loss = iou_utils.iou_loss(box_outputs, box_targets,
                                      self.iou_loss_type)
    box_iou_loss = tf.reduce_sum(box_iou_loss) / normalizer
    return box_iou_loss


class EfficientDetNetTrain(efficientdet_keras.EfficientDetNet):
  """A customized trainer for EfficientDet.

  see https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit
  """

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    log_dir = os.path.join(self.config.model_dir, 'train_images')
    self.summary_writer = tf.summary.create_file_writer(log_dir)

  def _freeze_vars(self):
    if self.config.var_freeze_expr:
      return [
          v for v in self.trainable_variables
          if not re.match(self.config.var_freeze_expr, v.name)
      ]
    return self.trainable_variables

  def _reg_l2_loss(self, weight_decay, regex=r'.*(kernel|weight):0$'):
    """Return regularization l2 loss loss."""
    var_match = re.compile(regex)
    return weight_decay * tf.add_n([
        tf.nn.l2_loss(v) for v in self._freeze_vars() if var_match.match(v.name)
    ])

  def _detection_loss(self, cls_outputs, box_outputs, labels, loss_vals):
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
      loss_vals: A dict of loss values.

    Returns:
      total_loss: an integer tensor representing total loss reducing from
        class and box losses from all levels.
      cls_loss: an integer tensor representing total class loss.
      box_loss: an integer tensor representing total box regression loss.
      box_iou_loss: an integer tensor representing total box iou loss.
    """
    # Sum all positives in a batch for normalization and avoid zero
    # num_positives_sum, which would lead to inf loss during training
    dtype = cls_outputs[0].dtype
    num_positives_sum = tf.reduce_sum(labels['mean_num_positives']) + 1.0
    positives_momentum = self.config.positives_momentum or 0
    if positives_momentum > 0:
      # normalize the num_positive_examples for training stability.
      moving_normalizer_var = tf.Variable(
          0.0,
          name='moving_normalizer',
          dtype=dtype,
          synchronization=tf.VariableSynchronization.ON_READ,
          trainable=False,
          aggregation=tf.VariableAggregation.MEAN)
      num_positives_sum = tf.keras.backend.moving_average_update(
          moving_normalizer_var,
          num_positives_sum,
          momentum=self.config.positives_momentum)
    elif positives_momentum < 0:
      num_positives_sum = utils.cross_replica_mean(num_positives_sum)
    num_positives_sum = tf.cast(num_positives_sum, dtype)
    levels = range(len(cls_outputs))
    cls_losses = []
    box_losses = []
    for level in levels:
      # Onehot encoding for classification labels.
      cls_targets_at_level = tf.one_hot(
          labels['cls_targets_%d' % (level + self.config.min_level)],
          self.config.num_classes,
          dtype=dtype)

      if self.config.data_format == 'channels_first':
        bs, _, width, height, _ = cls_targets_at_level.get_shape().as_list()
        cls_targets_at_level = tf.reshape(cls_targets_at_level,
                                          [bs, -1, width, height])
      else:
        bs, width, height, _, _ = cls_targets_at_level.get_shape().as_list()
        cls_targets_at_level = tf.reshape(cls_targets_at_level,
                                          [bs, width, height, -1])

      class_loss_layer = self.loss.get(FocalLoss.__name__, None)
      if class_loss_layer:
        cls_loss = class_loss_layer([num_positives_sum, cls_targets_at_level],
                                    cls_outputs[level])
        if self.config.data_format == 'channels_first':
          cls_loss = tf.reshape(
              cls_loss, [bs, -1, width, height, self.config.num_classes])
        else:
          cls_loss = tf.reshape(
              cls_loss, [bs, width, height, -1, self.config.num_classes])
        cls_loss *= tf.cast(
            tf.expand_dims(
                tf.not_equal(
                    labels['cls_targets_%d' % (level + self.config.min_level)],
                    -2), -1), dtype)
        cls_loss_sum = tf.reduce_sum(cls_loss)
        cls_losses.append(tf.cast(cls_loss_sum, dtype))

      if self.config.box_loss_weight and self.loss.get(BoxLoss.__name__, None):
        box_targets_at_level = (
            labels['box_targets_%d' % (level + self.config.min_level)])
        box_loss_layer = self.loss[BoxLoss.__name__]
        box_losses.append(
            box_loss_layer([num_positives_sum, box_targets_at_level],
                           box_outputs[level]))

    if self.config.iou_loss_type:
      box_outputs = tf.concat([tf.reshape(v, [-1, 4]) for v in box_outputs],
                              axis=0)
      box_targets = tf.concat([
          tf.reshape(labels['box_targets_%d' %
                            (level + self.config.min_level)], [-1, 4])
          for level in levels
      ],
                              axis=0)
      box_iou_loss_layer = self.loss[BoxIouLoss.__name__]
      box_iou_loss = box_iou_loss_layer([num_positives_sum, box_targets],
                                        box_outputs)
      loss_vals['box_iou_loss'] = box_iou_loss
    else:
      box_iou_loss = 0

    cls_loss = tf.add_n(cls_losses) if cls_losses else 0
    box_loss = tf.add_n(box_losses) if box_losses else 0
    total_loss = (
        cls_loss + self.config.box_loss_weight * box_loss +
        self.config.iou_loss_weight * box_iou_loss)
    loss_vals['det_loss'] = total_loss
    loss_vals['cls_loss'] = cls_loss
    loss_vals['box_loss'] = box_loss
    return total_loss

  def train_step(self, data):
    """Train step.

    Args:
      data: Tuple of (images, labels). Image tensor with shape [batch_size,
        height, width, 3]. The height and width are fixed and equal.Input labels
        in a dictionary. The labels include class targets and box targets which
        are dense label maps. The labels are generated from get_input_fn
        function in data/dataloader.py.

    Returns:
      A dict record loss info.
    """
    images, labels = data
    if self.config.img_summary_steps:
      with self.summary_writer.as_default():
        tf.summary.image('input_image', images)
    with tf.GradientTape() as tape:
      if len(self.config.heads) == 2:
        cls_outputs, box_outputs, seg_outputs = util_keras.fp16_to_fp32_nested(
            self(images, training=True))
        loss_dtype = cls_outputs[0].dtype
      elif 'object_detection' in self.config.heads:
        cls_outputs, box_outputs = util_keras.fp16_to_fp32_nested(
            self(images, training=True))
        loss_dtype = cls_outputs[0].dtype
      elif 'segmentation' in self.config.heads:
        seg_outputs, = util_keras.fp16_to_fp32_nested(
            self(images, training=True))
        loss_dtype = seg_outputs.dtype
      else:
        raise ValueError('No valid head found: {}'.format(self.config.heads))
      labels = util_keras.fp16_to_fp32_nested(labels)

      total_loss = 0
      loss_vals = {}
      if 'object_detection' in self.config.heads:
        det_loss = self._detection_loss(cls_outputs, box_outputs, labels,
                                        loss_vals)
        total_loss += det_loss
      if 'segmentation' in self.config.heads:
        seg_loss_layer = (
            self.loss[tf.keras.losses.SparseCategoricalCrossentropy.__name__])
        seg_loss = seg_loss_layer(labels['image_masks'], seg_outputs)
        total_loss += seg_loss
        loss_vals['seg_loss'] = seg_loss

      reg_l2_loss = self._reg_l2_loss(self.config.weight_decay)
      loss_vals['reg_l2_loss'] = reg_l2_loss
      total_loss += tf.cast(reg_l2_loss, loss_dtype)
      if isinstance(self.optimizer,
                    tf.keras.mixed_precision.experimental.LossScaleOptimizer):
        scaled_loss = self.optimizer.get_scaled_loss(total_loss)
      else:
        scaled_loss = total_loss
    loss_vals['loss'] = total_loss
    loss_vals['learning_rate'] = self.optimizer.learning_rate(
        self.optimizer.iterations)
    trainable_vars = self._freeze_vars()
    scaled_gradients = tape.gradient(scaled_loss, trainable_vars)
    if isinstance(self.optimizer,
                  tf.keras.mixed_precision.experimental.LossScaleOptimizer):
      gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
    else:
      gradients = scaled_gradients
    if self.config.clip_gradients_norm > 0:
      clip_norm = abs(self.config.clip_gradients_norm)
      gradients = [
          tf.clip_by_norm(g, clip_norm) if g is not None else None
          for g in gradients
      ]
      gradients, _ = tf.clip_by_global_norm(gradients, clip_norm)
      loss_vals['gradient_norm'] = tf.linalg.global_norm(gradients)
    self.optimizer.apply_gradients(zip(gradients, trainable_vars))
    return loss_vals

  def test_step(self, data):
    """Test step.

    Args:
      data: Tuple of (images, labels). Image tensor with shape [batch_size,
        height, width, 3]. The height and width are fixed and equal.Input labels
        in a dictionary. The labels include class targets and box targets which
        are dense label maps. The labels are generated from get_input_fn
        function in data/dataloader.py.

    Returns:
      A dict record loss info.
    """
    images, labels = data
    if len(self.config.heads) == 2:
      cls_outputs, box_outputs, seg_outputs = util_keras.fp16_to_fp32_nested(
          self(images, training=False))
      loss_dtype = cls_outputs[0].dtype
    elif 'object_detection' in self.config.heads:
      cls_outputs, box_outputs = util_keras.fp16_to_fp32_nested(
          self(images, training=False))
      loss_dtype = cls_outputs[0].dtype
    elif 'segmentation' in self.config.heads:
      seg_outputs, = util_keras.fp16_to_fp32_nested(
          self(images, training=False))
      loss_dtype = seg_outputs.dtype
    else:
      raise ValueError('No valid head found: {}'.format(self.config.heads))

    labels = util_keras.fp16_to_fp32_nested(labels)

    total_loss = 0
    loss_vals = {}
    if 'object_detection' in self.config.heads:
      det_loss = self._detection_loss(cls_outputs, box_outputs, labels,
                                      loss_vals)
      total_loss += det_loss
    if 'segmentation' in self.config.heads:
      seg_loss_layer = (
          self.loss[tf.keras.losses.SparseCategoricalCrossentropy.__name__])
      seg_loss = seg_loss_layer(labels['image_masks'], seg_outputs)
      total_loss += seg_loss
      loss_vals['seg_loss'] = seg_loss
    reg_l2_loss = self._reg_l2_loss(self.config.weight_decay)
    loss_vals['reg_l2_loss'] = reg_l2_loss
    loss_vals['loss'] = total_loss + tf.cast(reg_l2_loss, loss_dtype)
    return loss_vals


class EfficientDetNetTrainHub(EfficientDetNetTrain):
  """EfficientDetNetTrain for Hub module."""

  def __init__(self, config, hub_module_url, name=''):
    super(efficientdet_keras.EfficientDetNet, self).__init__(name=name)
    self.config = config
    self.hub_module_url = hub_module_url
    self.base_model = hub.KerasLayer(hub_module_url, trainable=True)

    # class/box output prediction network.
    num_anchors = len(config.aspect_ratios) * config.num_scales

    conv2d_layer = efficientdet_keras.ClassNet.conv2d_layer(
        config.separable_conv, config.data_format)
    self.classes = efficientdet_keras.ClassNet.classes_layer(
        conv2d_layer,
        config.num_classes,
        num_anchors,
        name='class_net/class-predict')

    self.boxes = efficientdet_keras.BoxNet.boxes_layer(
        config.separable_conv,
        num_anchors,
        config.data_format,
        name='box_net/box-predict')

    log_dir = os.path.join(self.config.model_dir, 'train_images')
    self.summary_writer = tf.summary.create_file_writer(log_dir)

  def call(self, inputs, training):
    cls_outputs, box_outputs = self.base_model(inputs, training=training)
    for i in range(self.config.max_level - self.config.min_level + 1):
      cls_outputs[i] = self.classes(cls_outputs[i])
      box_outputs[i] = self.boxes(box_outputs[i])
    return (cls_outputs, box_outputs)
