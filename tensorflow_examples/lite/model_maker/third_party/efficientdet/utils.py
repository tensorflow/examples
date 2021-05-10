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
"""Common utils."""
import contextlib
import os
from typing import Text, Tuple, Union
from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow.compat.v2 as tf2
from tensorflow.python.eager import tape as tape_lib  # pylint:disable=g-direct-tensorflow-import
from tensorflow.python.tpu import tpu_function  # pylint:disable=g-direct-tensorflow-import
# pylint: disable=logging-format-interpolation


def srelu_fn(x):
  """Smooth relu: a smooth version of relu."""
  with tf.name_scope('srelu'):
    beta = tf.Variable(20.0, name='srelu_beta', dtype=tf.float32)**2
    beta = tf.cast(beta**2, x.dtype)
    safe_log = tf.math.log(tf.where(x > 0., beta * x + 1., tf.ones_like(x)))
    return tf.where((x > 0.), x - (1. / beta) * safe_log, tf.zeros_like(x))


def activation_fn(features: tf.Tensor, act_type: Text):
  """Customized non-linear activation type."""
  if act_type in ('silu', 'swish'):
    return tf.nn.swish(features)
  elif act_type == 'swish_native':
    return features * tf.sigmoid(features)
  elif act_type == 'hswish':
    return features * tf.nn.relu6(features + 3) / 6
  elif act_type == 'relu':
    return tf.nn.relu(features)
  elif act_type == 'relu6':
    return tf.nn.relu6(features)
  elif act_type == 'mish':
    return features * tf.math.tanh(tf.math.softplus(features))
  elif act_type == 'srelu':
    return srelu_fn(features)
  else:
    raise ValueError('Unsupported act_type {}'.format(act_type))


def cross_replica_mean(t, num_shards_per_group=None):
  """Calculates the average value of input tensor across TPU replicas."""
  num_shards = tpu_function.get_tpu_context().number_of_shards
  if not num_shards_per_group:
    return tf.tpu.cross_replica_sum(t) / tf.cast(num_shards, t.dtype)

  group_assignment = None
  if num_shards_per_group > 1:
    if num_shards % num_shards_per_group != 0:
      raise ValueError('num_shards: %d mod shards_per_group: %d, should be 0' %
                       (num_shards, num_shards_per_group))
    num_groups = num_shards // num_shards_per_group
    group_assignment = [[
        x for x in range(num_shards) if x // num_shards_per_group == y
    ] for y in range(num_groups)]
  return tf.tpu.cross_replica_sum(t, group_assignment) / tf.cast(
      num_shards_per_group, t.dtype)


def get_ema_vars():
  """Get all exponential moving average (ema) variables."""
  ema_vars = tf.trainable_variables() + \
             tf.get_collection(tf.GraphKeys.MOVING_AVERAGE_VARIABLES)
  for v in tf.global_variables():
    # We maintain mva for batch norm moving mean and variance as well.
    if 'moving_mean' in v.name or 'moving_variance' in v.name:
      ema_vars.append(v)
  return list(set(ema_vars))


def get_ckpt_var_map(ckpt_path, ckpt_scope, var_scope, skip_mismatch=None):
  """Get a var map for restoring from pretrained checkpoints.

  Args:
    ckpt_path: string. A pretrained checkpoint path.
    ckpt_scope: string. Scope name for checkpoint variables.
    var_scope: string. Scope name for model variables.
    skip_mismatch: skip variables if shape mismatch.

  Returns:
    var_map: a dictionary from checkpoint name to model variables.
  """
  logging.info('Init model from checkpoint {}'.format(ckpt_path))
  if not ckpt_scope.endswith('/') or not var_scope.endswith('/'):
    raise ValueError('Please specific scope name ending with /')
  if ckpt_scope.startswith('/'):
    ckpt_scope = ckpt_scope[1:]
  if var_scope.startswith('/'):
    var_scope = var_scope[1:]

  var_map = {}
  # Get the list of vars to restore.
  model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=var_scope)
  reader = tf.train.load_checkpoint(ckpt_path)
  ckpt_var_name_to_shape = reader.get_variable_to_shape_map()
  ckpt_var_names = set(reader.get_variable_to_shape_map().keys())

  if tf.distribute.get_replica_context():
    replica_id = tf.get_static_value(
        tf.distribute.get_replica_context().replica_id_in_sync_group)
  else:
    replica_id = 0

  for i, v in enumerate(model_vars):
    var_op_name = v.op.name

    if replica_id >= 1:
      var_op_name = ''.join(var_op_name.rsplit(f'/replica_{replica_id}', 1))

    if not var_op_name.startswith(var_scope):
      logging.info('skip {} -- does not match scope {}'.format(
          var_op_name, var_scope))
    ckpt_var = ckpt_scope + var_op_name[len(var_scope):]

    if (ckpt_var not in ckpt_var_names and
        var_op_name.endswith('/ExponentialMovingAverage')):
      ckpt_var = ckpt_scope + var_op_name[:-len('/ExponentialMovingAverage')]

    if ckpt_var not in ckpt_var_names:
      if 'Momentum' in ckpt_var or 'RMSProp' in ckpt_var:
        # Skip optimizer variables.
        continue
      if skip_mismatch:
        logging.info('skip {} ({}) -- not in ckpt'.format(
            var_op_name, ckpt_var))
        continue
      raise ValueError('{} is not in ckpt {}'.format(v.op, ckpt_path))

    if v.shape != ckpt_var_name_to_shape[ckpt_var]:
      if skip_mismatch:
        logging.info('skip {} ({} vs {}) -- shape mismatch'.format(
            var_op_name, v.shape, ckpt_var_name_to_shape[ckpt_var]))
        continue
      raise ValueError('shape mismatch {} ({} vs {})'.format(
          var_op_name, v.shape, ckpt_var_name_to_shape[ckpt_var]))

    if i < 5:
      # Log the first few elements for sanity check.
      logging.info('Init {} from ckpt var {}'.format(var_op_name, ckpt_var))
    var_map[ckpt_var] = v

  return var_map


class TpuBatchNormalization(tf.keras.layers.BatchNormalization):
  """Cross replica batch normalization."""

  def __init__(self, fused=False, **kwargs):
    if not kwargs.get('name', None):
      kwargs['name'] = 'tpu_batch_normalization'
    if fused in (True, None):
      raise ValueError('TpuBatchNormalization does not support fused=True.')
    super().__init__(fused=fused, **kwargs)

  def _moments(self, inputs, reduction_axes, keep_dims):
    """Compute the mean and variance: it overrides the original _moments."""
    shard_mean, shard_variance = super()._moments(
        inputs, reduction_axes, keep_dims=keep_dims)

    num_shards = tpu_function.get_tpu_context().number_of_shards or 1
    num_shards_per_group = min(32, num_shards)  # aggregate up to 32 cores.
    logging.info('TpuBatchNormalization with num_shards_per_group {}'.format(
        num_shards_per_group))
    if num_shards_per_group > 1:
      # Compute variance using: Var[X]= E[X^2] - E[X]^2.
      shard_square_of_mean = tf.math.square(shard_mean)
      shard_mean_of_square = shard_variance + shard_square_of_mean
      group_mean = cross_replica_mean(shard_mean, num_shards_per_group)
      group_mean_of_square = cross_replica_mean(shard_mean_of_square,
                                                num_shards_per_group)
      group_variance = group_mean_of_square - tf.math.square(group_mean)
      return (group_mean, group_variance)
    else:
      return (shard_mean, shard_variance)

  def call(self, inputs, training=None):
    outputs = super().call(inputs, training)
    # A temporary hack for tf1 compatibility with keras batch norm.
    for u in self.updates:
      tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, u)
    return outputs


class SyncBatchNormalization(tf.keras.layers.BatchNormalization):
  """Cross replica batch normalization."""

  def __init__(self, fused=False, **kwargs):
    if not kwargs.get('name', None):
      kwargs['name'] = 'tpu_batch_normalization'
    if fused in (True, None):
      raise ValueError('SyncBatchNormalization does not support fused=True.')
    super().__init__(fused=fused, **kwargs)

  def _moments(self, inputs, reduction_axes, keep_dims):
    """Compute the mean and variance: it overrides the original _moments."""
    shard_mean, shard_variance = super()._moments(
        inputs, reduction_axes, keep_dims=keep_dims)

    replica_context = tf.distribute.get_replica_context()
    num_shards = replica_context.num_replicas_in_sync or 1

    if num_shards > 1:
      # Compute variance using: Var[X]= E[X^2] - E[X]^2.
      shard_square_of_mean = tf.math.square(shard_mean)
      shard_mean_of_square = shard_variance + shard_square_of_mean
      group_mean = replica_context.all_reduce(
          tf.distribute.ReduceOp.MEAN, shard_mean)
      group_mean_of_square = replica_context.all_reduce(
          tf.distribute.ReduceOp.MEAN, shard_mean_of_square)
      group_variance = group_mean_of_square - tf.math.square(group_mean)
      return (group_mean, group_variance)
    else:
      return (shard_mean, shard_variance)

  def call(self, inputs, training=None):
    outputs = super().call(inputs, training)
    # A temporary hack for tf1 compatibility with keras batch norm.
    for u in self.updates:
      tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, u)
    return outputs


class BatchNormalization(tf.keras.layers.BatchNormalization):
  """Fixed default name of BatchNormalization to match TpuBatchNormalization."""

  def __init__(self, **kwargs):
    if not kwargs.get('name', None):
      kwargs['name'] = 'tpu_batch_normalization'
    super().__init__(**kwargs)

  def call(self, inputs, training=None):
    outputs = super().call(inputs, training)
    # A temporary hack for tf1 compatibility with keras batch norm.
    for u in self.updates:
      tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, u)
    return outputs


def batch_norm_class(is_training, strategy=None):
  if is_training and strategy == 'tpu':
    return TpuBatchNormalization
  elif is_training and strategy == 'gpus':
    return SyncBatchNormalization
  else:
    return BatchNormalization


def batch_normalization(inputs, training=False, strategy=None, **kwargs):
  """A wrapper for TpuBatchNormalization."""
  bn_layer = batch_norm_class(training, strategy)(**kwargs)
  return bn_layer(inputs, training=training)


def batch_norm_act(inputs,
                   is_training_bn: bool,
                   act_type: Union[Text, None],
                   init_zero: bool = False,
                   data_format: Text = 'channels_last',
                   momentum: float = 0.99,
                   epsilon: float = 1e-3,
                   strategy: Text = None,
                   name: Text = None):
  """Performs a batch normalization followed by a non-linear activation.

  Args:
    inputs: `Tensor` of shape `[batch, channels, ...]`.
    is_training_bn: `bool` for whether the model is training.
    act_type: non-linear relu function type. If None, omits the relu operation.
    init_zero: `bool` if True, initializes scale parameter of batch
      normalization with 0 instead of 1 (default).
    data_format: `str` either "channels_first" for `[batch, channels, height,
      width]` or "channels_last for `[batch, height, width, channels]`.
    momentum: `float`, momentume of batch norm.
    epsilon: `float`, small value for numerical stability.
    strategy: string to specify training strategy for TPU/GPU/CPU.
    name: the name of the batch normalization layer

  Returns:
    A normalized `Tensor` with the same `data_format`.
  """
  if init_zero:
    gamma_initializer = tf.zeros_initializer()
  else:
    gamma_initializer = tf.ones_initializer()

  if data_format == 'channels_first':
    axis = 1
  else:
    axis = 3

  inputs = batch_normalization(
      inputs=inputs,
      axis=axis,
      momentum=momentum,
      epsilon=epsilon,
      center=True,
      scale=True,
      training=is_training_bn,
      strategy=strategy,
      gamma_initializer=gamma_initializer,
      name=name)

  if act_type:
    inputs = activation_fn(inputs, act_type)
  return inputs


def drop_connect(inputs, is_training, survival_prob):
  """Drop the entire conv with given survival probability."""
  # "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf
  if not is_training:
    return inputs

  # Compute tensor.
  batch_size = tf.shape(inputs)[0]
  random_tensor = survival_prob
  random_tensor += tf.random.uniform([batch_size, 1, 1, 1], dtype=inputs.dtype)
  binary_tensor = tf.floor(random_tensor)
  # Unlike conventional way that multiply survival_prob at test time, here we
  # divide survival_prob at training time, such that no addition compute is
  # needed at test time.
  output = inputs / survival_prob * binary_tensor
  return output


def num_params_flops(readable_format=True):
  """Return number of parameters and flops."""
  nparams = np.sum(
      [np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
  options = tf.profiler.ProfileOptionBuilder.float_operation()
  options['output'] = 'none'
  flops = tf.profiler.profile(
      tf.get_default_graph(), options=options).total_float_ops
  # We use flops to denote multiply-adds, which is counted as 2 ops in tfprof.
  flops = flops // 2
  if readable_format:
    nparams = float(nparams) * 1e-6
    flops = float(flops) * 1e-9
  return nparams, flops


conv_kernel_initializer = tf.initializers.variance_scaling()
dense_kernel_initializer = tf.initializers.variance_scaling()


class Pair(tuple):

  def __new__(cls, name, value):
    return super().__new__(cls, (name, value))

  def __init__(self, name, _):  # pylint: disable=super-init-not-called
    self.name = name


def scalar(name, tensor, is_tpu=True):
  """Stores a (name, Tensor) tuple in a custom collection."""
  logging.info('Adding scale summary {}'.format(Pair(name, tensor)))
  if is_tpu:
    tf.add_to_collection('scalar_summaries', Pair(name, tf.reduce_mean(tensor)))
  else:
    tf.summary.scalar(name, tf.reduce_mean(tensor))


def image(name, tensor, is_tpu=True):
  logging.info('Adding image summary {}'.format(Pair(name, tensor)))
  if is_tpu:
    tf.add_to_collection('image_summaries', Pair(name, tensor))
  else:
    tf.summary.image(name, tensor)


def get_tpu_host_call(global_step, params):
  """Get TPU host call for summaries."""
  scalar_summaries = tf.get_collection('scalar_summaries')
  if params['img_summary_steps']:
    image_summaries = tf.get_collection('image_summaries')
  else:
    image_summaries = []
  if not scalar_summaries and not image_summaries:
    return None  # No summaries to write.

  model_dir = params['model_dir']
  iterations_per_loop = params.get('iterations_per_loop', 100)
  img_steps = params['img_summary_steps']

  def host_call_fn(global_step, *args):
    """Training host call. Creates summaries for training metrics."""
    gs = global_step[0]
    with tf2.summary.create_file_writer(
        model_dir, max_queue=iterations_per_loop).as_default():
      with tf2.summary.record_if(True):
        for i, _ in enumerate(scalar_summaries):
          name = scalar_summaries[i][0]
          tensor = args[i][0]
          tf2.summary.scalar(name, tensor, step=gs)

      if img_steps:
        with tf2.summary.record_if(lambda: tf.math.equal(gs % img_steps, 0)):
          # Log images every 1k steps.
          for i, _ in enumerate(image_summaries):
            name = image_summaries[i][0]
            tensor = args[i + len(scalar_summaries)]
            tf2.summary.image(name, tensor, step=gs)

      return tf.summary.all_v2_summary_ops()

  reshaped_tensors = [tf.reshape(t, [1]) for _, t in scalar_summaries]
  reshaped_tensors += [t for _, t in image_summaries]
  global_step_t = tf.reshape(global_step, [1])
  return host_call_fn, [global_step_t] + reshaped_tensors


def archive_ckpt(ckpt_eval, ckpt_objective, ckpt_path):
  """Archive a checkpoint if the metric is better."""
  ckpt_dir, ckpt_name = os.path.split(ckpt_path)

  saved_objective_path = os.path.join(ckpt_dir, 'best_objective.txt')
  saved_objective = float('-inf')
  if tf.io.gfile.exists(saved_objective_path):
    with tf.io.gfile.GFile(saved_objective_path, 'r') as f:
      saved_objective = float(f.read())
  if saved_objective > ckpt_objective:
    logging.info('Ckpt {} is worse than {}'.format(ckpt_objective,
                                                   saved_objective))
    return False

  filenames = tf.io.gfile.glob(ckpt_path + '.*')
  if filenames is None:
    logging.info('No files to copy for checkpoint {}'.format(ckpt_path))
    return False

  # clear up the backup folder.
  backup_dir = os.path.join(ckpt_dir, 'backup')
  if tf.io.gfile.exists(backup_dir):
    tf.io.gfile.rmtree(backup_dir)

  # rename the old checkpoints to backup folder.
  dst_dir = os.path.join(ckpt_dir, 'archive')
  if tf.io.gfile.exists(dst_dir):
    logging.info('mv {} to {}'.format(dst_dir, backup_dir))
    tf.io.gfile.rename(dst_dir, backup_dir)

  # Write checkpoints.
  tf.io.gfile.makedirs(dst_dir)
  for f in filenames:
    dest = os.path.join(dst_dir, os.path.basename(f))
    tf.io.gfile.copy(f, dest, overwrite=True)
  ckpt_state = tf.train.generate_checkpoint_state_proto(
      dst_dir, model_checkpoint_path=os.path.join(dst_dir, ckpt_name))
  with tf.io.gfile.GFile(os.path.join(dst_dir, 'checkpoint'), 'w') as f:
    f.write(str(ckpt_state))
  with tf.io.gfile.GFile(os.path.join(dst_dir, 'best_eval.txt'), 'w') as f:
    f.write('%s' % ckpt_eval)

  # Update the best objective.
  with tf.io.gfile.GFile(saved_objective_path, 'w') as f:
    f.write('%f' % ckpt_objective)

  logging.info('Copying checkpoint {} to {}'.format(ckpt_path, dst_dir))
  return True


def parse_image_size(image_size: Union[Text, int, Tuple[int, int]]):
  """Parse the image size and return (height, width).

  Args:
    image_size: A integer, a tuple (H, W), or a string with HxW format.

  Returns:
    A tuple of integer (height, width).
  """
  if isinstance(image_size, int):
    # image_size is integer, with the same width and height.
    return (image_size, image_size)

  if isinstance(image_size, str):
    # image_size is a string with format WxH
    width, height = image_size.lower().split('x')
    return (int(height), int(width))

  if isinstance(image_size, tuple):
    return image_size

  raise ValueError('image_size must be an int, WxH string, or (height, width)'
                   'tuple. Was %r' % image_size)


def get_feat_sizes(image_size: Union[Text, int, Tuple[int, int]],
                   max_level: int):
  """Get feat widths and heights for all levels.

  Args:
    image_size: A integer, a tuple (H, W), or a string with HxW format.
    max_level: maximum feature level.

  Returns:
    feat_sizes: a list of tuples (height, width) for each level.
  """
  image_size = parse_image_size(image_size)
  feat_sizes = [{'height': image_size[0], 'width': image_size[1]}]
  feat_size = image_size
  for _ in range(1, max_level + 1):
    feat_size = ((feat_size[0] - 1) // 2 + 1, (feat_size[1] - 1) // 2 + 1)
    feat_sizes.append({'height': feat_size[0], 'width': feat_size[1]})
  return feat_sizes


def verify_feats_size(feats,
                      feat_sizes,
                      min_level,
                      max_level,
                      data_format='channels_last'):
  """Verify the feature map sizes."""
  expected_output_size = feat_sizes[min_level:max_level + 1]
  for cnt, size in enumerate(expected_output_size):
    h_id, w_id = (2, 3) if data_format == 'channels_first' else (1, 2)
    if feats[cnt].shape[h_id] != size['height']:
      raise ValueError(
          'feats[{}] has shape {} but its height should be {}.'
          '(input_height: {}, min_level: {}, max_level: {}.)'.format(
              cnt, feats[cnt].shape, size['height'], feat_sizes[0]['height'],
              min_level, max_level))
    if feats[cnt].shape[w_id] != size['width']:
      raise ValueError(
          'feats[{}] has shape {} but its width should be {}.'
          '(input_width: {}, min_level: {}, max_level: {}.)'.format(
              cnt, feats[cnt].shape, size['width'], feat_sizes[0]['width'],
              min_level, max_level))


def get_precision(strategy: str, mixed_precision: bool = False):
  """Get the precision policy for a given strategy."""
  if mixed_precision:
    if strategy == 'tpu':
      return 'mixed_bfloat16'

    if tf.config.list_physical_devices('GPU'):
      return 'mixed_float16'

    # TODO(fsx950223): Fix CPU float16 inference
    # https://github.com/google/automl/issues/504
    logging.warning('float16 is not supported for CPU, use float32 instead')
    return 'float32'

  return 'float32'


@contextlib.contextmanager
def float16_scope():
  """Scope class for float16."""

  def _custom_getter(getter, *args, **kwargs):
    """Returns a custom getter that methods must be called under."""
    cast_to_float16 = False
    requested_dtype = kwargs['dtype']
    if requested_dtype == tf.float16:
      kwargs['dtype'] = tf.float32
      cast_to_float16 = True
    var = getter(*args, **kwargs)
    if cast_to_float16:
      var = tf.cast(var, tf.float16)
    return var

  with tf.variable_scope('', custom_getter=_custom_getter) as varscope:
    yield varscope


def set_precision_policy(policy_name: Text = None):
  """Set precision policy according to the name.

  Args:
    policy_name: precision policy name, one of 'float32', 'mixed_float16',
      'mixed_bfloat16', or None.
  """
  if not policy_name:
    return

  assert policy_name in ('mixed_float16', 'mixed_bfloat16', 'float32')
  logging.info('use mixed precision policy name %s', policy_name)
  tf.compat.v1.keras.layers.enable_v2_dtype_behavior()
  # mixed_float16 training is not supported for now, so disable loss_scale.
  # float32 and mixed_bfloat16 do not need loss scale for training.
  policy = tf2.keras.mixed_precision.Policy(policy_name)
  tf2.keras.mixed_precision.set_global_policy(policy)


def build_model_with_precision(pp, mm, ii, *args, **kwargs):
  """Build model with its inputs/params for a specified precision context.

  This is highly specific to this codebase, and not intended to be general API.
  Advanced users only. DO NOT use it if you don't know what it does.
  NOTE: short argument names are intended to avoid conficts with kwargs.

  Args:
    pp: A string, precision policy name, such as "mixed_float16".
    mm: A function, for rmodel builder.
    ii: A tensor, for model inputs.
    *args: A list of model arguments.
    **kwargs: A dict, extra model parameters.

  Returns:
    the output of mm model.
  """
  if pp == 'mixed_bfloat16':
    set_precision_policy(pp)
    inputs = tf.cast(ii, tf.bfloat16)
    with tf.tpu.bfloat16_scope():
      outputs = mm(inputs, *args, **kwargs)
  elif pp == 'mixed_float16':
    set_precision_policy(pp)
    inputs = tf.cast(ii, tf.float16)
    with float16_scope():
      outputs = mm(inputs, *args, **kwargs)
  elif not pp or pp == 'float32':
    outputs = mm(ii, *args, **kwargs)
  else:
    raise ValueError('Unknow precision name {}'.format(pp))

  # Users are responsible to convert the dtype of all outputs.
  return outputs


def _recompute_grad(f):
  """An eager-compatible version of recompute_grad.

  For f(*args, **kwargs), this supports gradients with respect to args or
  kwargs, but kwargs are currently only supported in eager-mode.
  Note that for keras layer and model objects, this is handled automatically.

  Warning: If `f` was originally a tf.keras Model or Layer object, `g` will not
  be able to access the member variables of that object, because `g` returns
  through the wrapper function `inner`.  When recomputing gradients through
  objects that inherit from keras, we suggest keeping a reference to the
  underlying object around for the purpose of accessing these variables.

  Args:
    f: function `f(*x)` that returns a `Tensor` or sequence of `Tensor` outputs.

  Returns:
   A function `g` that wraps `f`, but which recomputes `f` on the backwards
   pass of a gradient call.
  """

  @tf.custom_gradient
  def inner(*args, **kwargs):
    """Inner function closure for calculating gradients."""
    current_var_scope = tf.get_variable_scope()
    with tape_lib.stop_recording():
      result = f(*args, **kwargs)

    def grad_wrapper(*wrapper_args, **grad_kwargs):
      """Wrapper function to accomodate lack of kwargs in graph mode decorator."""

      @tf.custom_gradient
      def inner_recompute_grad(*dresult):
        """Nested custom gradient function for computing grads in reverse and forward mode autodiff."""
        # Gradient calculation for reverse mode autodiff.
        variables = grad_kwargs.get('variables')
        with tf.GradientTape() as t:
          id_args = tf.nest.map_structure(tf.identity, args)
          t.watch(id_args)
          if variables is not None:
            t.watch(variables)
          with tf.control_dependencies(dresult):
            with tf.variable_scope(current_var_scope):
              result = f(*id_args, **kwargs)
        kw_vars = []
        if variables is not None:
          kw_vars = list(variables)
        grads = t.gradient(
            result,
            list(id_args) + kw_vars,
            output_gradients=dresult,
            unconnected_gradients=tf.UnconnectedGradients.ZERO)

        def transpose(*t_args, **t_kwargs):
          """Gradient function calculation for forward mode autodiff."""
          # Just throw an error since gradients / activations are not stored on
          # tape for recompute.
          raise NotImplementedError(
              'recompute_grad tried to transpose grad of {}. '
              'Consider not using recompute_grad in forward mode'
              'autodiff'.format(f.__name__))

        return (grads[:len(id_args)], grads[len(id_args):]), transpose

      return inner_recompute_grad(*wrapper_args)

    return result, grad_wrapper

  return inner


def recompute_grad(recompute=False):
  """Decorator determine whether use gradient checkpoint."""

  def _wrapper(f):
    if recompute:
      return _recompute_grad(f)
    return f

  return _wrapper
