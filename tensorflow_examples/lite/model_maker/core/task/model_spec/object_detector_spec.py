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
import functools
import os
import tempfile
from typing import Optional, Tuple, Dict

from absl import logging
import tensorflow as tf
from tensorflow_examples.lite.model_maker.core import compat
from tensorflow_examples.lite.model_maker.core.api.api_util import mm_export
from tensorflow_examples.lite.model_maker.core.data_util import object_detector_dataloader
from tensorflow_examples.lite.model_maker.core.task import configs
from tensorflow_examples.lite.model_maker.core.task.model_spec import util

from tensorflow_examples.lite.model_maker.third_party.efficientdet import coco_metric
from tensorflow_examples.lite.model_maker.third_party.efficientdet import hparams_config
from tensorflow_examples.lite.model_maker.third_party.efficientdet import utils
from tensorflow_examples.lite.model_maker.third_party.efficientdet.keras import efficientdet_keras
from tensorflow_examples.lite.model_maker.third_party.efficientdet.keras import eval_tflite
from tensorflow_examples.lite.model_maker.third_party.efficientdet.keras import label_util
from tensorflow_examples.lite.model_maker.third_party.efficientdet.keras import postprocess
from tensorflow_examples.lite.model_maker.third_party.efficientdet.keras import train
from tensorflow_examples.lite.model_maker.third_party.efficientdet.keras import train_lib

# Number of calibration steps for full integer quantization. 500 steps are
# enough to get a reasonable post-quantization result.
_NUM_CALIBRATION_STEPS = 500


def _get_ordered_label_map(
    label_map: Optional[Dict[int, str]]) -> Optional[Dict[int, str]]:
  """Gets label_map as an OrderedDict instance with ids sorted."""
  if not label_map:
    return label_map
  ordered_label_map = collections.OrderedDict()
  for idx in sorted(label_map.keys()):
    ordered_label_map[idx] = label_map[idx]
  return ordered_label_map


class ExportModel(efficientdet_keras.EfficientDetModel):
  """Model to be exported as SavedModel/TFLite format."""

  def __init__(self,
               model: tf.keras.Model,
               config: hparams_config.Config,
               pre_mode: Optional[str] = 'infer',
               post_mode: Optional[str] = 'global',
               name: Optional[str] = ''):
    """Initilizes an instance with the keras model and pre/post_mode paramaters.

    Args:
      model: The EfficientDetNet model used for training which doesn't have pre
        and post processing.
      config: Model configuration.
      pre_mode: Pre-processing Mode, must be {None, 'infer'}.
      post_mode: Post-processing Mode, must be {None, 'global', 'per_class',
        'tflite'}.
      name: Model name.
    """
    super(efficientdet_keras.EfficientDetNet, self).__init__(name=name)
    self.model = model
    self.config = config
    self.pre_mode = pre_mode
    self.post_mode = post_mode

  @tf.function
  def __call__(self, inputs: tf.Tensor):
    """Calls this model.

    Args:
      inputs: a tensor with common shape [batch, height, width, channels].

    Returns:
      the output tensor list.
    """
    config = self.config

    # Preprocess.
    inputs, scales = self._preprocessing(inputs, config.image_size,
                                         config.mean_rgb, config.stddev_rgb,
                                         self.pre_mode)
    # Network.
    outputs = self.model(inputs, training=False)

    # Postprocess for detection.
    det_outputs = self._postprocess(outputs[0], outputs[1], scales,
                                    self.post_mode)
    outputs = det_outputs + outputs[2:]

    return outputs


@mm_export('object_detector.EfficientDetSpec')
class EfficientDetModelSpec(object):
  """A specification of the EfficientDet model."""

  compat_tf_versions = compat.get_compat_tf_versions(2)

  def __init__(self,
               model_name: str,
               uri: str,
               hparams: str = '',
               model_dir: Optional[str] = None,
               epochs: int = 50,
               batch_size: int = 64,
               steps_per_execution: int = 1,
               moving_average_decay: int = 0,
               var_freeze_expr: str = '(efficientnet|fpn_cells|resample_p6)',
               tflite_max_detections: int = 25,
               strategy: Optional[str] = None,
               tpu: Optional[str] = None,
               gcp_project: Optional[str] = None,
               tpu_zone: Optional[str] = None,
               use_xla: bool = False,
               profile: bool = False,
               debug: bool = False,
               tf_random_seed: int = 111111,
               verbose: int = 0) -> None:
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
      tflite_max_detections: The max number of output detections in the TFLite
        model.
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
      verbose: verbosity mode for `tf.keras.callbacks.ModelCheckpoint`, 0 or 1.
    """
    self.model_name = model_name
    self.uri = uri
    self.batch_size = batch_size
    config = hparams_config.get_efficientdet_config(model_name)
    config.override(hparams)
    config.image_size = utils.parse_image_size(config.image_size)
    config.var_freeze_expr = var_freeze_expr
    config.moving_average_decay = moving_average_decay
    config.tflite_max_detections = tflite_max_detections
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
        debug=debug,
        verbose=verbose)
    config.override(params, True)
    self.config = config

    # set mixed precision policy by keras api.
    precision = utils.get_precision(config.strategy, config.mixed_precision)
    policy = tf.keras.mixed_precision.experimental.Policy(precision)
    tf.keras.mixed_precision.experimental.set_policy(policy)

  def create_model(self) -> tf.keras.Model:
    """Creates the EfficientDet model."""
    return train_lib.EfficientDetNetTrainHub(
        config=self.config, hub_module_url=self.uri)

  def train(self,
            model: tf.keras.Model,
            train_dataset: tf.data.Dataset,
            steps_per_epoch: int,
            val_dataset: Optional[tf.data.Dataset],
            validation_steps: int,
            epochs: Optional[int] = None,
            batch_size: Optional[int] = None,
            val_json_file: Optional[str] = None) -> tf.keras.Model:
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

  def _get_evaluator_and_label_map(
      self, json_file: str
  ) -> Tuple[coco_metric.EvaluationMetric, Optional[Dict[int, str]]]:
    """Gets evaluator and label_map for evaluation."""
    label_map = label_util.get_label_map(self.config.label_map)
    # Sorts label_map.keys since pycocotools.cocoeval uses sorted catIds
    # (category ids) in COCOeval class.
    label_map = _get_ordered_label_map(label_map)

    evaluator = coco_metric.EvaluationMetric(
        filename=json_file, label_map=label_map)

    evaluator.reset_states()
    return evaluator, label_map

  def _get_metric_dict(self, evaluator: coco_metric.EvaluationMetric,
                       label_map: collections.OrderedDict) -> Dict[str, float]:
    """Gets the metric dict for evaluation."""
    metrics = evaluator.result(log_level=tf.compat.v1.logging.INFO)
    metric_dict = {}
    for i, name in enumerate(evaluator.metric_names):
      metric_dict[name] = metrics[i]

    if label_map:
      for i, cid in enumerate(label_map.keys()):
        name = 'AP_/%s' % label_map[cid]
        metric_dict[name] = metrics[i + len(evaluator.metric_names)]
    return metric_dict

  def evaluate(self,
               model: tf.keras.Model,
               dataset: tf.data.Dataset,
               steps: int,
               json_file: str = None) -> Dict[str, float]:
    """Evaluate the EfficientDet keras model.

    Args:
      model: The keras model to be evaluated.
      dataset: tf.data.Dataset used for evaluation.
      steps: Number of steps to evaluate the model.
      json_file: JSON with COCO data format containing golden bounding boxes.
        Used for validation. If None, use the ground truth from the dataloader.
        Refer to
        https://towardsdatascience.com/coco-data-format-for-object-detection-a4c5eaf518c5
          for the description of COCO data format.

    Returns:
      A dict contains AP metrics.
    """
    evaluator, label_map = self._get_evaluator_and_label_map(json_file)
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
    progbar = tf.keras.utils.Progbar(steps)
    for i, (images, labels) in enumerate(dataset):
      self.ds_strategy.run(_get_detections, (images, labels))
      progbar.update(i)
    print()

    metric_dict = self._get_metric_dict(evaluator, label_map)
    return metric_dict

  def evaluate_tflite(self,
                      tflite_filepath: str,
                      dataset: tf.data.Dataset,
                      steps: int,
                      json_file: str = None) -> Dict[str, float]:
    """Evaluate the EfficientDet TFLite model.

    Args:
      tflite_filepath: File path to the TFLite model.
      dataset: tf.data.Dataset used for evaluation.
      steps: Number of steps to evaluate the model.
      json_file: JSON with COCO data format containing golden bounding boxes.
        Used for validation. If None, use the ground truth from the dataloader.
        Refer to
        https://towardsdatascience.com/coco-data-format-for-object-detection-a4c5eaf518c5
          for the description of COCO data format.

    Returns:
      A dict contains AP metrics.
    """
    # TODO(b/182441458): Use the task library for evaluation instead once it
    # supports python interface.
    evaluator, label_map = self._get_evaluator_and_label_map(json_file)
    dataset = dataset.take(steps)

    lite_runner = eval_tflite.LiteRunner(tflite_filepath, only_network=False)
    progbar = tf.keras.utils.Progbar(steps)
    for i, (images, labels) in enumerate(dataset):
      # Get the output result after post-processing NMS op.
      nms_boxes, nms_classes, nms_scores, _ = lite_runner.run(images)

      # CLASS_OFFSET is used since label_id for `background` is 0 in label_map
      # while it's not actually included the model. We don't need to add the
      # offset in the Android application.
      nms_classes += postprocess.CLASS_OFFSET

      height, width = utils.parse_image_size(self.config.image_size)
      normalize_factor = tf.constant([height, width, height, width],
                                     dtype=tf.float32)
      nms_boxes *= normalize_factor
      if labels['image_scales'] is not None:
        scales = tf.expand_dims(tf.expand_dims(labels['image_scales'], -1), -1)
        nms_boxes = nms_boxes * tf.cast(scales, nms_boxes.dtype)
      detections = postprocess.generate_detections_from_nms_output(
          nms_boxes, nms_classes, nms_scores, labels['source_ids'])

      detections = postprocess.transform_detections(detections)
      evaluator.update_state(labels['groundtruth_data'].numpy(),
                             detections.numpy())
      progbar.update(i)

    metric_dict = self._get_metric_dict(evaluator, label_map)
    return metric_dict

  def export_saved_model(self,
                         model: tf.keras.Model,
                         saved_model_dir: str,
                         batch_size: Optional[int] = None,
                         pre_mode: Optional[str] = 'infer',
                         post_mode: Optional[str] = 'global') -> None:
    """Saves the model to Tensorflow SavedModel.

    Args:
      model: The EfficientDetNet model used for training which doesn't have pre
        and post processing.
      saved_model_dir: Folder path for saved model.
      batch_size: Batch size to be saved in saved_model.
      pre_mode: Pre-processing Mode in ExportModel, must be {None, 'infer'}.
      post_mode: Post-processing Mode in ExportModel, must be {None, 'global',
        'per_class', 'tflite'}.
    """
    config = self.config
    # Sets the keras model optimizer to None when exporting to saved model.
    # Otherwise, it fails with `NotImplementedError`: "Learning rate schedule
    # must override get_config".
    original_optimizer = model.optimizer
    model.optimizer = None
    # Creates ExportModel which has pre and post processing.
    export_model = ExportModel(model, config, pre_mode, post_mode)

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

    tf.saved_model.save(
        export_model,
        saved_model_dir,
        signatures=export_model.__call__.get_concrete_function(input_spec))
    model.optimizer = original_optimizer

  def get_default_quantization_config(
      self, representative_data: object_detector_dataloader.DataLoader
  ) -> configs.QuantizationConfig:
    """Gets the default quantization configuration."""
    # Sets `inference_output_type=None` since the output op is the custom NMS op
    # which can't be quantized so that the tflite model output should be float.
    # `supported_ops` contains both `TFLITE_BUILTINS_INT8` and `TFLITE_BUILTINS`
    # due to the same reason: custom NMS op should be included as the
    # `TFLITE_BUILTINS` op.
    config = configs.QuantizationConfig.for_int8(
        representative_data,
        inference_output_type=None,
        supported_ops=[
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.TFLITE_BUILTINS
        ])
    config.experimental_new_quantizer = True
    return config

  def export_tflite(
      self,
      model: tf.keras.Model,
      tflite_filepath: str,
      quantization_config: Optional[configs.QuantizationConfig] = None) -> None:
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
      model: The EfficientDetNet model used for training which doesn't have pre
        and post processing.
      tflite_filepath: File path to save tflite model.
      quantization_config: Configuration for post-training quantization.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
      self.export_saved_model(
          model, temp_dir, batch_size=1, pre_mode=None, post_mode='tflite')
      converter = tf.lite.TFLiteConverter.from_saved_model(temp_dir)

      if quantization_config:
        converter = quantization_config.get_converter_with_quantization(
            converter, model_spec=self)

        # TFLITE_BUILTINS is needed for TFLite's custom NMS op for integer only
        # quantization.
        if tf.lite.OpsSet.TFLITE_BUILTINS not in converter.target_spec.supported_ops:
          converter.target_spec.supported_ops += [
              tf.lite.OpsSet.TFLITE_BUILTINS
          ]

      tflite_model = converter.convert()

      with tf.io.gfile.GFile(tflite_filepath, 'wb') as f:
        f.write(tflite_model)


efficientdet_lite0_spec = functools.partial(
    EfficientDetModelSpec,
    model_name='efficientdet-lite0',
    uri='https://tfhub.dev/tensorflow/efficientdet/lite0/feature-vector/1',
)
efficientdet_lite0_spec.__doc__ = util.wrap_doc(
    EfficientDetModelSpec,
    'Creates EfficientDet-Lite0 model spec. See also: `tflite_model_maker.object_detector.EfficientDetSpec`.'
)
mm_export('object_detector.EfficientDetLite0Spec').export_constant(
    __name__, 'efficientdet_lite0_spec')

efficientdet_lite1_spec = functools.partial(
    EfficientDetModelSpec,
    model_name='efficientdet-lite1',
    uri='https://tfhub.dev/tensorflow/efficientdet/lite1/feature-vector/1',
)
efficientdet_lite1_spec.__doc__ = util.wrap_doc(
    EfficientDetModelSpec,
    'Creates EfficientDet-Lite1 model spec. See also: `tflite_model_maker.object_detector.EfficientDetSpec`.'
)
mm_export('object_detector.EfficientDetLite1Spec').export_constant(
    __name__, 'efficientdet_lite1_spec')

efficientdet_lite2_spec = functools.partial(
    EfficientDetModelSpec,
    model_name='efficientdet-lite2',
    uri='https://tfhub.dev/tensorflow/efficientdet/lite2/feature-vector/1',
)
efficientdet_lite2_spec.__doc__ = util.wrap_doc(
    EfficientDetModelSpec,
    'Creates EfficientDet-Lite2 model spec. See also: `tflite_model_maker.object_detector.EfficientDetSpec`.'
)
mm_export('object_detector.EfficientDetLite2Spec').export_constant(
    __name__, 'efficientdet_lite2_spec')

efficientdet_lite3_spec = functools.partial(
    EfficientDetModelSpec,
    model_name='efficientdet-lite3',
    uri='https://tfhub.dev/tensorflow/efficientdet/lite3/feature-vector/1',
)
efficientdet_lite3_spec.__doc__ = util.wrap_doc(
    EfficientDetModelSpec,
    'Creates EfficientDet-Lite3 model spec. See also: `tflite_model_maker.object_detector.EfficientDetSpec`.'
)
mm_export('object_detector.EfficientDetLite3Spec').export_constant(
    __name__, 'efficientdet_lite3_spec')

efficientdet_lite4_spec = functools.partial(
    EfficientDetModelSpec,
    model_name='efficientdet-lite4',
    uri='https://tfhub.dev/tensorflow/efficientdet/lite4/feature-vector/2',
)
efficientdet_lite4_spec.__doc__ = util.wrap_doc(
    EfficientDetModelSpec,
    'Creates EfficientDet-Lite4 model spec. See also: `tflite_model_maker.object_detector.EfficientDetSpec`.'
)
mm_export('object_detector.EfficientDetLite4Spec').export_constant(
    __name__, 'efficientdet_lite4_spec')
