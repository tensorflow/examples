# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Model specification for object detection."""

import collections
import os
import tempfile

from absl import logging
import tensorflow as tf
from tensorflow_examples.lite.model_maker.core import compat
from tensorflow_examples.lite.model_maker.core.task.model_spec import util

from tensorflow_examples.lite.model_maker.third_party.efficientdet import coco_metric
from tensorflow_examples.lite.model_maker.third_party.efficientdet import hparams_config
from tensorflow_examples.lite.model_maker.third_party.efficientdet import utils
from tensorflow_examples.lite.model_maker.third_party.efficientdet.keras import efficientdet_keras
from tensorflow_examples.lite.model_maker.third_party.efficientdet.keras import inference
from tensorflow_examples.lite.model_maker.third_party.efficientdet.keras import label_util
from tensorflow_examples.lite.model_maker.third_party.efficientdet.keras import postprocess
from tensorflow_examples.lite.model_maker.third_party.efficientdet.keras import train
from tensorflow_examples.lite.model_maker.third_party.efficientdet.keras import train_lib
from tensorflow_examples.lite.model_maker.third_party.efficientdet.keras import util_keras


def _get_ordered_label_map(label_map):
  """Gets label_map as an OrderedDict instance with ids sorted."""
  if not label_map:
    return label_map
  ordered_label_map = collections.OrderedDict()
  for idx in sorted(label_map.keys()):
    ordered_label_map[idx] = label_map[idx]
  return ordered_label_map


class EfficientDetModelSpec(object):
  """A specification of the EfficientDet model."""

  compat_tf_versions = compat.get_compat_tf_versions(2)

  def __init__(self,
               model_name,
               uri,
               hparams='',
               model_dir=None,
               epochs=50,
               batch_size=64,
               steps_per_execution=1,
               moving_average_decay=0,
               var_freeze_expr='(efficientnet|fpn_cells|resample_p6)',
               strategy=None,
               tpu=None,
               gcp_project=None,
               tpu_zone=None,
               use_xla=False,
               profile=False,
               debug=False,
               tf_random_seed=111111):
    """Initialze an instance with model paramaters.

    Args:
      model_name: Model name.
      uri: TF-Hub path/url to EfficientDet module.
      hparams: Hyperparameters used to overwrite default configuration. Can be
        1) Dict, contains parameter names and values; 2) String, Comma separated
        k=v pairs of hyperparameters; 3) String, yaml filename which's a module
        containing attributes to use as hyperparameters.
      model_dir: The location to save the model checkpoint files.
      epochs: Default training epochs.
      batch_size: Training & Evaluation batch size.
      steps_per_execution: Number of steps per training execution.
      moving_average_decay: Float. The decay to use for maintaining moving
        averages of the trained parameters.
      var_freeze_expr: Expression to freeze variables.
      strategy:  A string specifying which distribution strategy to use.
        Accepted values are 'tpu', 'gpus', None. tpu' means to use TPUStrategy.
        'gpus' mean to use MirroredStrategy for multi-gpus. If None, use TF
        default with OneDeviceStrategy.
      tpu: The Cloud TPU to use for training. This should be either the name
        used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470
          url.
      gcp_project: Project name for the Cloud TPU-enabled project. If not
        specified, we will attempt to automatically detect the GCE project from
        metadata.
      tpu_zone: GCE zone where the Cloud TPU is located in. If not specified, we
        will attempt to automatically detect the GCE project from metadata.
      use_xla: Use XLA even if strategy is not tpu. If strategy is tpu, always
        use XLA, and this flag has no effect.
      profile: Enable profile mode.
      debug: Enable debug mode.
      tf_random_seed: Fixed random seed for deterministic execution across runs
        for debugging.
    """
    self.model_name = model_name
    self.uri = uri
    self.batch_size = batch_size
    config = hparams_config.get_efficientdet_config(model_name)
    config.override(hparams)
    config.image_size = utils.parse_image_size(config.image_size)
    config.var_freeze_expr = var_freeze_expr
    config.moving_average_decay = moving_average_decay
    if epochs:
      config.num_epochs = epochs

    if use_xla and strategy != 'tpu':
      tf.config.optimizer.set_jit(True)
      for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    if debug:
      tf.config.experimental_run_functions_eagerly(True)
      tf.debugging.set_log_device_placement(True)
      os.environ['TF_DETERMINISTIC_OPS'] = '1'
      tf.random.set_seed(tf_random_seed)
      logging.set_verbosity(logging.DEBUG)

    if strategy == 'tpu':
      tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
          tpu, zone=tpu_zone, project=gcp_project)
      tf.config.experimental_connect_to_cluster(tpu_cluster_resolver)
      tf.tpu.experimental.initialize_tpu_system(tpu_cluster_resolver)
      ds_strategy = tf.distribute.TPUStrategy(tpu_cluster_resolver)
      logging.info('All devices: %s', tf.config.list_logical_devices('TPU'))
      tf.config.set_soft_device_placement(True)
    elif strategy == 'gpus':
      ds_strategy = tf.distribute.MirroredStrategy()
      logging.info('All devices: %s', tf.config.list_physical_devices('GPU'))
    else:
      if tf.config.list_physical_devices('GPU'):
        ds_strategy = tf.distribute.OneDeviceStrategy('device:GPU:0')
      else:
        ds_strategy = tf.distribute.OneDeviceStrategy('device:CPU:0')

    self.ds_strategy = ds_strategy

    if model_dir is None:
      model_dir = tempfile.mkdtemp()
    params = dict(
        profile=profile,
        model_name=model_name,
        steps_per_execution=steps_per_execution,
        model_dir=model_dir,
        strategy=strategy,
        batch_size=batch_size,
        tf_random_seed=tf_random_seed,
        debug=debug)
    config.override(params, True)
    self.config = config

    # set mixed precision policy by keras api.
    precision = utils.get_precision(config.strategy, config.mixed_precision)
    policy = tf.keras.mixed_precision.experimental.Policy(precision)
    tf.keras.mixed_precision.experimental.set_policy(policy)

  def create_model(self):
    """Creates the EfficientDet model."""
    return train_lib.EfficientDetNetTrainHub(
        config=self.config, hub_module_url=self.uri)

  def train(self,
            model,
            train_dataset,
            steps_per_epoch,
            val_dataset,
            validation_steps,
            epochs=None,
            batch_size=None,
            val_json_file=None):
    """Run EfficientDet training."""
    config = self.config
    if not epochs:
      epochs = config.num_epochs

    if not batch_size:
      batch_size = config.batch_size

    config.update(
        dict(
            steps_per_epoch=steps_per_epoch,
            eval_samples=batch_size * validation_steps,
            val_json_file=val_json_file,
            batch_size=batch_size))
    train.setup_model(model, config)
    train.init_experimental(config)
    model.fit(
        train_dataset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        callbacks=train_lib.get_callbacks(config.as_dict(), val_dataset),
        validation_data=val_dataset,
        validation_steps=validation_steps)
    return model

  def evaluate(self, model, dataset, steps, json_file=None):
    """Evaluate the EfficientDet keras model."""
    label_map = label_util.get_label_map(self.config.label_map)
    # Sorts label_map.keys since pycocotools.cocoeval uses sorted catIds
    # (category ids) in COCOeval class.
    label_map = _get_ordered_label_map(label_map)

    evaluator = coco_metric.EvaluationMetric(
        filename=json_file, label_map=label_map)

    evaluator.reset_states()
    dataset = dataset.take(steps)

    @tf.function
    def _get_detections(images, labels):
      cls_outputs, box_outputs = model(images, training=False)
      detections = postprocess.generate_detections(self.config, cls_outputs,
                                                   box_outputs,
                                                   labels['image_scales'],
                                                   labels['source_ids'])
      tf.numpy_function(evaluator.update_state, [
          labels['groundtruth_data'],
          postprocess.transform_detections(detections)
      ], [])

    dataset = self.ds_strategy.experimental_distribute_dataset(dataset)
    for (images, labels) in dataset:
      self.ds_strategy.run(_get_detections, (images, labels))

    metrics = evaluator.result()
    metric_dict = {}
    for i, name in enumerate(evaluator.metric_names):
      metric_dict[name] = metrics[i]

    if label_map:
      for i, cid in enumerate(label_map.keys()):
        name = 'AP_/%s' % label_map[cid]
        metric_dict[name] = metrics[i + len(evaluator.metric_names)]
    return metric_dict

  def export_saved_model(self,
                         saved_model_dir,
                         batch_size=None,
                         pre_mode='infer',
                         post_mode='global'):
    """Saves the model to Tensorflow SavedModel.

    Args:
      saved_model_dir: Folder path for saved model.
      batch_size: Batch size to be saved in saved_model.
      pre_mode: Pre-processing Mode in ExportModel, must be {None, 'infer'}.
      post_mode: Post-processing Mode in ExportModel, must be {None, 'global',
        'per_class'}.
    """
    # Create EfficientDetModel with latest checkpoint.
    config = self.config
    tf.keras.backend.clear_session()
    model = efficientdet_keras.EfficientDetModel(config=config)
    model.build((batch_size, *config.image_size, 3))
    if config.model_dir:
      util_keras.restore_ckpt(
          model,
          config.model_dir,
          config['moving_average_decay'],
          skip_mismatch=False)
    else:
      # EfficientDetModel is random initialized without restoring the
      # checkpoint. This is mainly used in object_detector_test and shouldn't be
      #  used if we want to export trained model.
      tf.compat.v1.logging.warn('Need to restore the checkpoint for '
                                'EfficientDet.')
    # Gets tf.TensorSpec.
    if pre_mode is None:
      # Input is the preprocessed image that's already resized to a certain
      # input shape.
      input_spec = tf.TensorSpec(
          shape=[batch_size, *config.image_size, 3],
          dtype=tf.float32,
          name='images')
    else:
      # Input is that raw image that can be in any input shape,
      input_spec = tf.TensorSpec(
          shape=[batch_size, None, None, 3], dtype=tf.uint8, name='images')

    export_model = inference.ExportModel(
        model, pre_mode=pre_mode, post_mode=post_mode)
    tf.saved_model.save(
        export_model,
        saved_model_dir,
        signatures=export_model.__call__.get_concrete_function(input_spec))

  def export_tflite(self, tflite_filepath, quantization_config=None):
    """Converts the retrained model to tflite format and saves it.

    The exported TFLite model has the following inputs & outputs:
    One input:
      image: a float32 tensor of shape[1, height, width, 3] containing the
        normalized input image. `self.config.image_size` is [height, width].

    Four Outputs:
      detection_boxes: a float32 tensor of shape [1, num_boxes, 4] with box
        locations.
      detection_classes: a float32 tensor of shape [1, num_boxes] with class
        indices.
      detection_scores: a float32 tensor of shape [1, num_boxes] with class
        scores.
      num_boxes: a float32 tensor of size 1 containing the number of detected
        boxes.

    Args:
      tflite_filepath: File path to save tflite model.
      quantization_config: Configuration for post-training quantization.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
      self.export_saved_model(
          temp_dir, batch_size=1, pre_mode=None, post_mode='tflite')
      converter = tf.lite.TFLiteConverter.from_saved_model(temp_dir)
      if quantization_config:
        converter = quantization_config.get_converter_with_quantization(
            converter, model_spec=self)

      # TFLITE_BUILTINS is needed for TFLite's custom NMS op for integer only
      # quantization.
      if tf.lite.OpsSet.TFLITE_BUILTINS not in converter.target_spec.supported_ops:
        converter.target_spec.supported_ops += [tf.lite.OpsSet.TFLITE_BUILTINS]
      tflite_model = converter.convert()

      with tf.io.gfile.GFile(tflite_filepath, 'wb') as f:
        f.write(tflite_model)


def efficientdet_lite0_spec(**kwargs):
  args = util.dict_with_default(
      default_dict=dict(
          model_name='efficientdet-lite0',
          uri='https://tfhub.dev/tensorflow/efficientdet/lite0/feature-vector/1'
      ),
      **kwargs)
  return EfficientDetModelSpec(**args)


def efficientdet_lite1_spec(**kwargs):
  args = util.dict_with_default(
      default_dict=dict(
          model_name='efficientdet-lite1',
          uri='https://tfhub.dev/tensorflow/efficientdet/lite1/feature-vector/1'
      ),
      **kwargs)
  return EfficientDetModelSpec(**args)


def efficientdet_lite2_spec(**kwargs):
  args = util.dict_with_default(
      default_dict=dict(
          model_name='efficientdet-lite2',
          uri='https://tfhub.dev/tensorflow/efficientdet/lite2/feature-vector/1'
      ),
      **kwargs)
  return EfficientDetModelSpec(**args)


def efficientdet_lite3_spec(**kwargs):
  args = util.dict_with_default(
      default_dict=dict(
          model_name='efficientdet-lite3',
          uri='https://tfhub.dev/tensorflow/efficientdet/lite3/feature-vector/1'
      ),
      **kwargs)
  return EfficientDetModelSpec(**args)


def efficientdet_lite4_spec(**kwargs):
  args = util.dict_with_default(
      default_dict=dict(
          model_name='efficientdet-lite4',
          uri='https://tfhub.dev/tensorflow/efficientdet/lite4/feature-vector/1'
      ),
      **kwargs)
  return EfficientDetModelSpec(**args)
