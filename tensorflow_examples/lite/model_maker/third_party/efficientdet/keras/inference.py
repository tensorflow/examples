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
r"""Inference related utilities."""
import copy
import os
import time
from typing import Text, Dict, Any
from absl import logging
import numpy as np
import tensorflow as tf

from tensorflow_examples.lite.model_maker.third_party.efficientdet import dataloader
from tensorflow_examples.lite.model_maker.third_party.efficientdet import hparams_config
from tensorflow_examples.lite.model_maker.third_party.efficientdet import utils
from tensorflow_examples.lite.model_maker.third_party.efficientdet.keras import efficientdet_keras
from tensorflow_examples.lite.model_maker.third_party.efficientdet.keras import label_util
from tensorflow_examples.lite.model_maker.third_party.efficientdet.keras import util_keras
from tensorflow_examples.lite.model_maker.third_party.efficientdet.visualize import vis_utils


def visualize_image(image,
                    boxes,
                    classes,
                    scores,
                    label_map=None,
                    min_score_thresh=0.01,
                    max_boxes_to_draw=1000,
                    line_thickness=2,
                    **kwargs):
  """Visualizes a given image.

  Args:
    image: a image with shape [H, W, C].
    boxes: a box prediction with shape [N, 4] ordered [ymin, xmin, ymax, xmax].
    classes: a class prediction with shape [N].
    scores: A list of float value with shape [N].
    label_map: a dictionary from class id to name.
    min_score_thresh: minimal score for showing. If claass probability is below
      this threshold, then the object will not show up.
    max_boxes_to_draw: maximum bounding box to draw.
    line_thickness: how thick is the bounding box line.
    **kwargs: extra parameters.

  Returns:
    output_image: an output image with annotated boxes and classes.
  """
  label_map = label_util.get_label_map(label_map or 'coco')
  category_index = {k: {'id': k, 'name': label_map[k]} for k in label_map}
  img = np.array(image)
  vis_utils.visualize_boxes_and_labels_on_image_array(
      img,
      boxes,
      classes,
      scores,
      category_index,
      min_score_thresh=min_score_thresh,
      max_boxes_to_draw=max_boxes_to_draw,
      line_thickness=line_thickness,
      **kwargs)
  return img


class ExportNetwork(tf.Module):

  def __init__(self, model):
    super().__init__()
    self.model = model

  @tf.function
  def __call__(self, imgs):
    return tf.nest.flatten(self.model(imgs, training=False))


class ExportModel(tf.Module):
  """Model to be exported as SavedModel/TFLite format."""

  def __init__(self, model, pre_mode='infer', post_mode='global'):
    super().__init__()
    self.model = model
    self.pre_mode = pre_mode
    self.post_mode = post_mode

  @tf.function
  def __call__(self, imgs):
    return self.model(
        imgs, training=False, pre_mode=self.pre_mode, post_mode=self.post_mode)


class ServingDriver:
  """A driver for serving single or batch images.

  This driver supports serving with image files or arrays, with configurable
  batch size.

  Example 1. Serving streaming image contents:

    driver = inference.ServingDriver(
      'efficientdet-d0', '/tmp/efficientdet-d0', batch_size=1)
    driver.build()
    for m in image_iterator():
      predictions = driver.serve_files([m])
      boxes, scores, classes, _ = tf.nest.map_structure(np.array, predictions)
      driver.visualize(m, boxes[0], scores[0], classes[0])
      # m is the new image with annotated boxes.

  Example 2. Serving batch image contents:

    imgs = []
    for f in ['/tmp/1.jpg', '/tmp/2.jpg']:
      imgs.append(np.array(Image.open(f)))

    driver = inference.ServingDriver(
      'efficientdet-d0', '/tmp/efficientdet-d0', batch_size=len(imgs))
    driver.build()
    predictions = driver.serve(imgs)
    boxes, scores, classes, _ = tf.nest.map_structure(np.array, predictions)
    for i in range(len(imgs)):
      driver.visualize(imgs[i], boxes[i], scores[i], classes[i])

  Example 3: another way is to use SavedModel:

    # step1: export a model.
    driver = inference.ServingDriver('efficientdet-d0', '/tmp/efficientdet-d0')
    driver.build()
    driver.export('/tmp/saved_model_path')

    # step2: Serve a model.
    driver.load(self.saved_model_dir)
    raw_images = []
    for f in tf.io.gfile.glob('/tmp/images/*.jpg'):
      raw_images.append(np.array(PIL.Image.open(f)))
    detections = driver.serve(raw_images)
    boxes, scores, classes, _ = tf.nest.map_structure(np.array, detections)
    for i in range(len(imgs)):
      driver.visualize(imgs[i], boxes[i], scores[i], classes[i])
  """

  def __init__(self,
               model_name: Text,
               ckpt_path: Text = None,
               batch_size: int = 1,
               only_network: bool = False,
               model_params: Dict[Text, Any] = None):
    """Initialize the inference driver.

    Args:
      model_name: target model name, such as efficientdet-d0.
      ckpt_path: checkpoint path, such as /tmp/efficientdet-d0/.
      batch_size: batch size for inference.
      only_network: only use the network without pre/post processing.
      model_params: model parameters for overriding the config.
    """
    super().__init__()
    self.model_name = model_name
    self.ckpt_path = ckpt_path
    self.batch_size = batch_size
    self.only_network = only_network

    self.params = hparams_config.get_detection_config(model_name).as_dict()

    if model_params:
      self.params.update(model_params)
    self.params.update(dict(is_training_bn=False))
    self.label_map = self.params.get('label_map', None)

    self._model = None

    mixed_precision = self.params.get('mixed_precision', None)
    precision = utils.get_precision(
        self.params.get('strategy', None), mixed_precision)
    policy = tf.keras.mixed_precision.experimental.Policy(precision)
    tf.keras.mixed_precision.experimental.set_policy(policy)

  @property
  def model(self):
    if not self._model:
      self.build()
    return self._model

  @model.setter
  def model(self, model):
    self._model = model

  def build(self, params_override=None):
    """Build model and restore checkpoints."""
    params = copy.deepcopy(self.params)
    if params_override:
      params.update(params_override)
    config = hparams_config.get_efficientdet_config(self.model_name)
    config.override(params)
    if self.only_network:
      self.model = efficientdet_keras.EfficientDetNet(config=config)
    else:
      self.model = efficientdet_keras.EfficientDetModel(config=config)
    image_size = utils.parse_image_size(params['image_size'])
    self.model.build((self.batch_size, *image_size, 3))
    util_keras.restore_ckpt(self.model, self.ckpt_path,
                            self.params['moving_average_decay'],
                            skip_mismatch=False)

  def visualize(self, image, boxes, classes, scores, **kwargs):
    """Visualize prediction on image."""
    return visualize_image(image, boxes, classes.astype(int), scores,
                           self.label_map, **kwargs)

  def benchmark(self, image_arrays, bm_runs=10, trace_filename=None):
    """Benchmark inference latency/throughput.

    Args:
      image_arrays: a list of images in numpy array format.
      bm_runs: Number of benchmark runs.
      trace_filename: If None, specify the filename for saving trace.
    """
    _, spec = self._get_model_and_spec()

    @tf.function(input_signature=[spec])
    def test_func(image_arrays):
      return self.model(image_arrays)  # pylint: disable=not-callable

    for _ in range(3):  # warmup 3 runs.
      test_func(image_arrays)

    start = time.perf_counter()
    for _ in range(bm_runs):
      test_func(image_arrays)
    end = time.perf_counter()
    inference_time = (end - start) / bm_runs

    print('Per batch inference time: ', inference_time)
    print('FPS: ', self.batch_size / inference_time)

    if trace_filename:
      options = tf.profiler.experimental.ProfilerOptions()
      tf.profiler.experimental.start(trace_filename, options)
      test_func(image_arrays)
      tf.profiler.experimental.stop()

  def serve(self, image_arrays):
    """Serve a list of image arrays.

    Args:
      image_arrays: A list of image content with each image has shape [height,
        width, 3] and uint8 type.

    Returns:
      A list of detections.
    """
    if isinstance(self.model, tf.lite.Interpreter):
      input_details = self.model.get_input_details()
      output_details = self.model.get_output_details()
      self.model.set_tensor(input_details[0]['index'], np.array(image_arrays))
      self.model.invoke()
      return [self.model.get_tensor(x['index']) for x in output_details]
    return self.model(image_arrays)  # pylint: disable=not-callable

  def load(self, saved_model_dir_or_frozen_graph: Text):
    """Load the model using saved model or a frozen graph."""
    # Load saved model if it is a folder.
    if tf.saved_model.contains_saved_model(saved_model_dir_or_frozen_graph):
      self.model = tf.saved_model.load(saved_model_dir_or_frozen_graph)
      return

    if saved_model_dir_or_frozen_graph.endswith('.tflite'):
      self.model = tf.lite.Interpreter(saved_model_dir_or_frozen_graph)
      self.model.allocate_tensors()
      return

    # Load a frozen graph.
    def wrap_frozen_graph(graph_def, inputs, outputs):
      # https://www.tensorflow.org/guide/migrate
      imports_graph_def_fn = lambda: tf.import_graph_def(graph_def, name='')
      wrapped_import = tf.compat.v1.wrap_function(imports_graph_def_fn, [])
      import_graph = wrapped_import.graph
      return wrapped_import.prune(
          tf.nest.map_structure(import_graph.as_graph_element, inputs),
          tf.nest.map_structure(import_graph.as_graph_element, outputs))

    graph_def = tf.Graph().as_graph_def()
    with tf.io.gfile.GFile(saved_model_dir_or_frozen_graph, 'rb') as f:
      graph_def.ParseFromString(f.read())

    self.model = wrap_frozen_graph(
        graph_def,
        inputs='images:0',
        outputs=['Identity:0', 'Identity_1:0', 'Identity_2:0', 'Identity_3:0'])

  def freeze(self, func):
    """Freeze the graph."""
    # pylint: disable=g-import-not-at-top,disable=g-direct-tensorflow-import
    from tensorflow.python.framework.convert_to_constants \
      import convert_variables_to_constants_v2_as_graph
    _, graphdef = convert_variables_to_constants_v2_as_graph(func)
    return graphdef

  def _get_model_and_spec(self, tflite=None):
    """Get model instance and export spec."""
    if self.only_network or tflite:
      image_size = utils.parse_image_size(self.params['image_size'])
      spec = tf.TensorSpec(
          shape=[self.batch_size, *image_size, 3],
          dtype=tf.float32,
          name='images')
      if self.only_network:
        export_model = ExportNetwork(self.model)
      else:
        # If export tflite, we should remove preprocessing since TFLite doesn't
        # support dynamic shape.
        logging.info('Export model without preprocessing.')
        # This section is only used for TFLite, so we use the applicable
        # pre_ & post_ modes.
        export_model = ExportModel(
            self.model, pre_mode=None, post_mode='tflite')
      return export_model, spec
    else:
      spec = tf.TensorSpec(
          shape=[self.batch_size, None, None, 3], dtype=tf.uint8, name='images')
      export_model = ExportModel(self.model)
      return export_model, spec

  def export(self,
             output_dir: Text = None,
             tensorrt: Text = None,
             tflite: Text = None,
             file_pattern: Text = None,
             num_calibration_steps: int = 2000):
    """Export a saved model, frozen graph, and potential tflite/tensorrt model.

    Args:
      output_dir: the output folder for saved model.
      tensorrt: If not None, must be {'FP32', 'FP16', 'INT8'}.
      tflite: Type for post-training quantization.
      file_pattern: Glob for tfrecords, e.g. coco/val-*.tfrecord.
      num_calibration_steps: Number of post-training quantization calibration
        steps to run.
    """
    export_model, input_spec = self._get_model_and_spec(tflite)
    image_size = utils.parse_image_size(self.params['image_size'])
    if output_dir:
      tf.saved_model.save(
          export_model,
          output_dir,
          signatures=export_model.__call__.get_concrete_function(input_spec))
      logging.info('Model saved at %s', output_dir)

      # also save freeze pb file.
      graphdef = self.freeze(
          export_model.__call__.get_concrete_function(input_spec))
      proto_path = tf.io.write_graph(
          graphdef, output_dir, self.model_name + '_frozen.pb', as_text=False)
      logging.info('Frozen graph saved at %s', proto_path)

    if tflite:
      shape = (self.batch_size, *image_size, 3)
      input_spec = tf.TensorSpec(
          shape=shape, dtype=input_spec.dtype, name=input_spec.name)
      # from_saved_model supports advanced converter features like op fusing.
      converter = tf.lite.TFLiteConverter.from_saved_model(output_dir)
      if tflite == 'FP32':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float32]
      elif tflite == 'FP16':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
      elif tflite == 'INT8':
        # Enables MLIR-based post-training quantization.
        converter.experimental_new_quantizer = True
        if file_pattern:
          config = hparams_config.get_efficientdet_config(self.model_name)
          config.override(self.params)
          ds = dataloader.InputReader(
              file_pattern,
              is_training=False,
              max_instances_per_image=config.max_instances_per_image)(
                  config, batch_size=self.batch_size)

          def representative_dataset_gen():
            for image, _ in ds.take(num_calibration_steps):
              yield [image]
        else:  # Used for debugging, can remove later.
          logging.warn('Use real representative dataset instead of fake ones.')
          num_calibration_steps = 10
          def representative_dataset_gen():  # rewrite this for real data.
            for _ in range(num_calibration_steps):
              yield [tf.ones(shape, dtype=input_spec.dtype)]

        converter.representative_dataset = representative_dataset_gen
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.inference_input_type = tf.uint8
        # TFLite's custom NMS op isn't supported by post-training quant,
        # so we add TFLITE_BUILTINS as well.
        supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.TFLITE_BUILTINS
        ]
        converter.target_spec.supported_ops = supported_ops

      else:
        raise ValueError(f'Invalid tflite {tflite}: must be FP32, FP16, INT8.')

      tflite_path = os.path.join(output_dir, tflite.lower() + '.tflite')
      tflite_model = converter.convert()
      tf.io.gfile.GFile(tflite_path, 'wb').write(tflite_model)
      logging.info('TFLite is saved at %s', tflite_path)

    if tensorrt:
      trt_path = os.path.join(output_dir, 'tensorrt_' + tensorrt.lower())
      conversion_params = tf.experimental.tensorrt.ConversionParams(
          max_workspace_size_bytes=(2 << 20),
          maximum_cached_engines=1,
          precision_mode=tensorrt.upper())
      converter = tf.experimental.tensorrt.Converter(
          output_dir, conversion_params=conversion_params)
      converter.convert()
      converter.save(trt_path)
      logging.info('TensorRT model is saved at %s', trt_path)
