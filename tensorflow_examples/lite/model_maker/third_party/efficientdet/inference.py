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
import functools
import os
import time
from typing import Text, Dict, Any, List, Tuple, Union
from absl import logging
import numpy as np
from PIL import Image
import tensorflow.compat.v1 as tf

from tensorflow_examples.lite.model_maker.third_party.efficientdet import dataloader
from tensorflow_examples.lite.model_maker.third_party.efficientdet import det_model_fn
from tensorflow_examples.lite.model_maker.third_party.efficientdet import hparams_config
from tensorflow_examples.lite.model_maker.third_party.efficientdet import utils
from tensorflow_examples.lite.model_maker.third_party.efficientdet.keras import efficientdet_keras
from tensorflow_examples.lite.model_maker.third_party.efficientdet.keras import label_util
from tensorflow_examples.lite.model_maker.third_party.efficientdet.keras import postprocess
from tensorflow_examples.lite.model_maker.third_party.efficientdet.visualize import vis_utils
from tensorflow.python.client import timeline  # pylint: disable=g-direct-tensorflow-import


def image_preprocess(image, image_size, mean_rgb, stddev_rgb):
  """Preprocess image for inference.

  Args:
    image: input image, can be a tensor or a numpy arary.
    image_size: single integer of image size for square image or tuple of two
      integers, in the format of (image_height, image_width).
    mean_rgb: Mean value of RGB, can be a list of float or a float value.
    stddev_rgb: Standard deviation of RGB, can be a list of float or a float
      value.

  Returns:
    (image, scale): a tuple of processed image and its scale.
  """
  input_processor = dataloader.DetectionInputProcessor(image, image_size)
  input_processor.normalize_image(mean_rgb, stddev_rgb)
  input_processor.set_scale_factors_to_output_size()
  image = input_processor.resize_and_crop_image()
  image_scale = input_processor.image_scale_to_original
  return image, image_scale


@tf.autograph.to_graph
def batch_image_files_decode(image_files):
  raw_images = tf.TensorArray(tf.uint8, size=0, dynamic_size=True)
  for i in tf.range(tf.shape(image_files)[0]):
    image = tf.io.decode_image(image_files[i])
    image.set_shape([None, None, None])
    raw_images = raw_images.write(i, image)
  return raw_images.stack()


def batch_image_preprocess(raw_images,
                           image_size: Union[int, Tuple[int, int]],
                           mean_rgb,
                           stddev_rgb,
                           batch_size: int = None):
  """Preprocess batched images for inference.

  Args:
    raw_images: a list of images, each image can be a tensor or a numpy arary.
    image_size: single integer of image size for square image or tuple of two
      integers, in the format of (image_height, image_width).
    mean_rgb: Mean value of RGB, can be a list of float or a float value.
    stddev_rgb: Standard deviation of RGB, can be a list of float or a float
      value.
    batch_size: if None, use map_fn to deal with dynamic batch size.

  Returns:
    (image, scale): a tuple of processed images and scales.
  """
  if not batch_size:
    # map_fn is a little bit slower due to some extra overhead.
    map_fn = functools.partial(
        image_preprocess,
        image_size=image_size,
        mean_rgb=mean_rgb,
        stddev_rgb=stddev_rgb)
    images, scales = tf.map_fn(
        map_fn, raw_images, dtype=(tf.float32, tf.float32), back_prop=False)
    return (images, scales)

  # If batch size is known, use a simple loop.
  scales, images = [], []
  for i in range(batch_size):
    image, scale = image_preprocess(raw_images[i], image_size, mean_rgb,
                                    stddev_rgb)
    scales.append(scale)
    images.append(image)
  images = tf.stack(images)
  scales = tf.stack(scales)
  return (images, scales)


def build_inputs(
    image_path_pattern: Text,
    image_size: Union[int, Tuple[int, int]],
    mean_rgb,
    stddev_rgb,
):
  """Read and preprocess input images.

  Args:
    image_path_pattern: a path to indicate a single or multiple files.
    image_size: single integer of image size for square image or tuple of two
      integers, in the format of (image_height, image_width).
    mean_rgb: Mean value of RGB, can be a list of float or a float value.
    stddev_rgb: Standard deviation of RGB, can be a list of float or a float
      value.

  Returns:
    (raw_images, images, scales): raw images, processed images, and scales.

  Raises:
    ValueError if image_path_pattern doesn't match any file.
  """
  raw_images, images, scales = [], [], []
  for f in tf.io.gfile.glob(image_path_pattern):
    image = Image.open(f)
    raw_images.append(image)
    image, scale = image_preprocess(image, image_size, mean_rgb, stddev_rgb)
    images.append(image)
    scales.append(scale)
  if not images:
    raise ValueError(
        'Cannot find any images for pattern {}'.format(image_path_pattern))
  return raw_images, tf.stack(images), tf.stack(scales)


def build_model(model_name: Text, inputs: tf.Tensor, **kwargs):
  """Build model for a given model name.

  Args:
    model_name: the name of the model.
    inputs: an image tensor or a numpy array.
    **kwargs: extra parameters for model builder.

  Returns:
    (cls_outputs, box_outputs): the outputs for class and box predictions.
    Each is a dictionary with key as feature level and value as predictions.
  """
  mixed_precision = kwargs.get('mixed_precision', None)
  precision = utils.get_precision(kwargs.get('strategy', None), mixed_precision)

  if kwargs.get('use_keras_model', None):

    def model_arch(feats, model_name=None, **kwargs):
      """Construct a model arch for keras models."""
      config = hparams_config.get_efficientdet_config(model_name)
      config.override(kwargs)
      model = efficientdet_keras.EfficientDetNet(config=config)
      cls_out_list, box_out_list = model(feats, training=False)
      # convert the list of model outputs to a dictionary with key=level.
      assert len(cls_out_list) == config.max_level - config.min_level + 1
      assert len(box_out_list) == config.max_level - config.min_level + 1
      cls_outputs, box_outputs = {}, {}
      for i in range(config.min_level, config.max_level + 1):
        cls_outputs[i] = cls_out_list[i - config.min_level]
        box_outputs[i] = box_out_list[i - config.min_level]
      return cls_outputs, box_outputs

  else:
    model_arch = det_model_fn.get_model_arch(model_name)

  cls_outputs, box_outputs = utils.build_model_with_precision(
      precision, model_arch, inputs, model_name, **kwargs)

  if mixed_precision:
    # Post-processing has multiple places with hard-coded float32.
    # TODO(tanmingxing): Remove them once post-process can adpat to dtypes.
    cls_outputs = {k: tf.cast(v, tf.float32) for k, v in cls_outputs.items()}
    box_outputs = {k: tf.cast(v, tf.float32) for k, v in box_outputs.items()}

  return cls_outputs, box_outputs


def restore_ckpt(sess, ckpt_path, ema_decay=0.9998, export_ckpt=None):
  """Restore variables from a given checkpoint.

  Args:
    sess: a tf session for restoring or exporting models.
    ckpt_path: the path of the checkpoint. Can be a file path or a folder path.
    ema_decay: ema decay rate. If None or zero or negative value, disable ema.
    export_ckpt: whether to export the restored model.
  """
  sess.run(tf.global_variables_initializer())
  if tf.io.gfile.isdir(ckpt_path):
    ckpt_path = tf.train.latest_checkpoint(ckpt_path)
  if ema_decay > 0:
    ema = tf.train.ExponentialMovingAverage(decay=0.0)
    ema_vars = utils.get_ema_vars()
    var_dict = ema.variables_to_restore(ema_vars)
    ema_assign_op = ema.apply(ema_vars)
  else:
    var_dict = utils.get_ema_vars()
    ema_assign_op = None

  tf.train.get_or_create_global_step()
  sess.run(tf.global_variables_initializer())
  saver = tf.train.Saver(var_dict, max_to_keep=1)
  if ckpt_path == '_':
    logging.info('Running test: do not load any ckpt.')
    return

  # Restore all variables from ckpt.
  saver.restore(sess, ckpt_path)

  if export_ckpt:
    print('export model to {}'.format(export_ckpt))
    if ema_assign_op is not None:
      sess.run(ema_assign_op)
    saver = tf.train.Saver(max_to_keep=1, save_relative_paths=True)
    saver.save(sess, export_ckpt)


def det_post_process(params: Dict[Any, Any], cls_outputs: Dict[int, tf.Tensor],
                     box_outputs: Dict[int, tf.Tensor], scales: List[float]):
  """Post preprocessing the box/class predictions.

  Args:
    params: a parameter dictionary that includes `min_level`, `max_level`,
      `batch_size`, and `num_classes`.
    cls_outputs: an OrderDict with keys representing levels and values
      representing logits in [batch_size, height, width, num_anchors].
    box_outputs: an OrderDict with keys representing levels and values
      representing box regression targets in [batch_size, height, width,
      num_anchors * 4].
    scales: a list of float values indicating image scale.

  Returns:
    detections_batch: a batch of detection results. Each detection is a tensor
      with each row as [image_id, ymin, xmin, ymax, xmax, score, class].
  """
  if params.get('combined_nms', None):
    # Use combined version for dynamic batch size.
    nms_boxes, nms_scores, nms_classes, _ = postprocess.postprocess_combined(
        params, cls_outputs, box_outputs, scales)
  else:
    nms_boxes, nms_scores, nms_classes, _ = postprocess.postprocess_global(
        params, cls_outputs, box_outputs, scales)

  batch_size = tf.shape(cls_outputs[params['min_level']])[0]
  img_ids = tf.expand_dims(
      tf.cast(tf.range(0, batch_size), nms_scores.dtype), -1)
  detections = [
      img_ids * tf.ones_like(nms_scores),
      nms_boxes[:, :, 0],
      nms_boxes[:, :, 1],
      nms_boxes[:, :, 2],
      nms_boxes[:, :, 3],
      nms_scores,
      nms_classes,
  ]
  return tf.stack(detections, axis=-1, name='detections')


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


def visualize_image_prediction(image,
                               prediction,
                               label_map=None,
                               **kwargs):
  """Viusalize detections on a given image.

  Args:
    image: Image content in shape of [height, width, 3].
    prediction: a list of vector, with each vector has the format of [image_id,
      ymin, xmin, ymax, xmax, score, class].
    label_map: a map from label id to name.
    **kwargs: extra parameters for vistualization, such as min_score_thresh,
      max_boxes_to_draw, and line_thickness.

  Returns:
    a list of annotated images.
  """
  boxes = prediction[:, 1:5]
  classes = prediction[:, 6].astype(int)
  scores = prediction[:, 5]

  return visualize_image(image, boxes, classes, scores, label_map, **kwargs)


class ServingDriver(object):
  """A driver for serving single or batch images.

  This driver supports serving with image files or arrays, with configurable
  batch size.

  Example 1. Serving streaming image contents:

    driver = inference.ServingDriver(
      'efficientdet-d0', '/tmp/efficientdet-d0', batch_size=1)
    driver.build()
    for m in image_iterator():
      predictions = driver.serve_files([m])
      driver.visualize(m, predictions[0])
      # m is the new image with annotated boxes.

  Example 2. Serving batch image contents:

    imgs = []
    for f in ['/tmp/1.jpg', '/tmp/2.jpg']:
      imgs.append(np.array(Image.open(f)))

    driver = inference.ServingDriver(
      'efficientdet-d0', '/tmp/efficientdet-d0', batch_size=len(imgs))
    driver.build()
    predictions = driver.serve_images(imgs)
    for i in range(len(imgs)):
      driver.visualize(imgs[i], predictions[i])

  Example 3: another way is to use SavedModel:

    # step1: export a model.
    driver = inference.ServingDriver('efficientdet-d0', '/tmp/efficientdet-d0')
    driver.build()
    driver.export('/tmp/saved_model_path')

    # step2: Serve a model.
    with tf.Session() as sess:
      tf.saved_model.load(sess, ['serve'], self.saved_model_dir)
      raw_images = []
      for f in tf.io.gfile.glob('/tmp/images/*.jpg'):
        raw_images.append(np.array(PIL.Image.open(f)))
      detections = sess.run('detections:0', {'image_arrays:0': raw_images})
      driver = inference.ServingDriver(
        'efficientdet-d0', '/tmp/efficientdet-d0')
      driver.visualize(raw_images[0], detections[0])
      PIL.Image.fromarray(raw_images[0]).save(output_image_path)
  """

  def __init__(self,
               model_name: Text,
               ckpt_path: Text,
               batch_size: int = 1,
               use_xla: bool = False,
               min_score_thresh: float = None,
               max_boxes_to_draw: float = None,
               line_thickness: int = None,
               model_params: Dict[Text, Any] = None):
    """Initialize the inference driver.

    Args:
      model_name: target model name, such as efficientdet-d0.
      ckpt_path: checkpoint path, such as /tmp/efficientdet-d0/.
      batch_size: batch size for inference.
      use_xla: Whether run with xla optimization.
      min_score_thresh: minimal score threshold for filtering predictions.
      max_boxes_to_draw: the maximum number of boxes per image.
      line_thickness: the line thickness for drawing boxes.
      model_params: model parameters for overriding the config.
    """
    self.model_name = model_name
    self.ckpt_path = ckpt_path
    self.batch_size = batch_size

    self.params = hparams_config.get_detection_config(model_name).as_dict()

    if model_params:
      self.params.update(model_params)
    self.params.update(dict(is_training_bn=False))
    self.label_map = self.params.get('label_map', None)

    self.signitures = None
    self.sess = None
    self.use_xla = use_xla

    self.min_score_thresh = min_score_thresh
    self.max_boxes_to_draw = max_boxes_to_draw
    self.line_thickness = line_thickness

  def __del__(self):
    if self.sess:
      self.sess.close()

  def _build_session(self):
    sess_config = tf.ConfigProto()
    if self.use_xla:
      sess_config.graph_options.optimizer_options.global_jit_level = (
          tf.OptimizerOptions.ON_2)
    return tf.Session(config=sess_config)

  def build(self, params_override=None):
    """Build model and restore checkpoints."""
    params = copy.deepcopy(self.params)
    if params_override:
      params.update(params_override)

    if not self.sess:
      self.sess = self._build_session()
    with self.sess.graph.as_default():
      image_files = tf.placeholder(tf.string, name='image_files', shape=[None])
      raw_images = batch_image_files_decode(image_files)
      raw_images = tf.identity(raw_images, name='image_arrays')
      images, scales = batch_image_preprocess(raw_images, params['image_size'],
                                              params['mean_rgb'],
                                              params['stddev_rgb'],
                                              self.batch_size)
      if params['data_format'] == 'channels_first':
        images = tf.transpose(images, [0, 3, 1, 2])
      class_outputs, box_outputs = build_model(self.model_name, images,
                                               **params)
      params.update(dict(batch_size=self.batch_size))
      detections = det_post_process(params, class_outputs, box_outputs, scales)

      restore_ckpt(
          self.sess,
          self.ckpt_path,
          ema_decay=self.params['moving_average_decay'],
          export_ckpt=None)

    self.signitures = {
        'image_files': image_files,
        'image_arrays': raw_images,
        'prediction': detections,
    }
    return self.signitures

  def visualize(self, image, prediction, **kwargs):
    """Visualize prediction on image."""
    return visualize_image_prediction(
        image,
        prediction,
        label_map=self.label_map,
        **kwargs)

  def serve_files(self, image_files: List[Text]):
    """Serve a list of input image files.

    Args:
      image_files: a list of image files with shape [1] and type string.

    Returns:
      A list of detections.
    """
    if not self.sess:
      self.build()
    predictions = self.sess.run(
        self.signitures['prediction'],
        feed_dict={self.signitures['image_files']: image_files})
    return predictions

  def benchmark(self, image_arrays, trace_filename=None):
    """Benchmark inference latency/throughput.

    Args:
      image_arrays: a list of images in numpy array format.
      trace_filename: If None, specify the filename for saving trace.
    """
    if not self.sess:
      self.build()

    # init session
    self.sess.run(
        self.signitures['prediction'],
        feed_dict={self.signitures['image_arrays']: image_arrays})

    start = time.perf_counter()
    for _ in range(10):
      self.sess.run(
          self.signitures['prediction'],
          feed_dict={self.signitures['image_arrays']: image_arrays})
    end = time.perf_counter()
    inference_time = (end - start) / 10

    print('Per batch inference time: ', inference_time)
    print('FPS: ', self.batch_size / inference_time)

    if trace_filename:
      run_options = tf.RunOptions()
      run_options.trace_level = tf.RunOptions.FULL_TRACE
      run_metadata = tf.RunMetadata()
      self.sess.run(
          self.signitures['prediction'],
          feed_dict={self.signitures['image_arrays']: image_arrays},
          options=run_options,
          run_metadata=run_metadata)
      with tf.io.gfile.GFile(trace_filename, 'w') as trace_file:
        trace = timeline.Timeline(step_stats=run_metadata.step_stats)
        trace_file.write(trace.generate_chrome_trace_format(show_memory=True))

  def serve_images(self, image_arrays):
    """Serve a list of image arrays.

    Args:
      image_arrays: A list of image content with each image has shape [height,
        width, 3] and uint8 type.

    Returns:
      A list of detections.
    """
    if not self.sess:
      self.build()
    predictions = self.sess.run(
        self.signitures['prediction'],
        feed_dict={self.signitures['image_arrays']: image_arrays})
    return predictions

  def load(self, saved_model_dir_or_frozen_graph: Text):
    """Load the model using saved model or a frozen graph."""
    if not self.sess:
      self.sess = self._build_session()
    self.signitures = {
        'image_files': 'image_files:0',
        'image_arrays': 'image_arrays:0',
        'prediction': 'detections:0',
    }

    # Load saved model if it is a folder.
    if tf.io.gfile.isdir(saved_model_dir_or_frozen_graph):
      return tf.saved_model.load(self.sess, ['serve'],
                                 saved_model_dir_or_frozen_graph)

    # Load a frozen graph.
    graph_def = tf.GraphDef()
    with tf.gfile.GFile(saved_model_dir_or_frozen_graph, 'rb') as f:
      graph_def.ParseFromString(f.read())
    return tf.import_graph_def(graph_def, name='')

  def freeze(self):
    """Freeze the graph."""
    output_names = [self.signitures['prediction'].op.name]
    graphdef = tf.graph_util.convert_variables_to_constants(
        self.sess, self.sess.graph_def, output_names)
    return graphdef

  def export(self,
             output_dir: Text,
             tflite_path: Text = None,
             tensorrt: Text = None):
    """Export a saved model, frozen graph, and potential tflite/tensorrt model.

    Args:
      output_dir: the output folder for saved model.
      tflite_path: the path for saved tflite file.
      tensorrt: If not None, must be {'FP32', 'FP16', 'INT8'}.
    """
    signitures = self.signitures
    signature_def_map = {
        'serving_default':
            tf.saved_model.predict_signature_def(
                {signitures['image_arrays'].name: signitures['image_arrays']},
                {signitures['prediction'].name: signitures['prediction']}),
    }
    b = tf.saved_model.Builder(output_dir)
    b.add_meta_graph_and_variables(
        self.sess,
        tags=['serve'],
        signature_def_map=signature_def_map,
        assets_collection=tf.get_collection(tf.GraphKeys.ASSET_FILEPATHS),
        clear_devices=True)
    b.save()
    logging.info('Model saved at %s', output_dir)

    # also save freeze pb file.
    graphdef = self.freeze()
    pb_path = os.path.join(output_dir, self.model_name + '_frozen.pb')
    tf.io.gfile.GFile(pb_path, 'wb').write(graphdef.SerializeToString())
    logging.info('Frozen graph saved at %s', pb_path)

    if tflite_path:
      height, width = utils.parse_image_size(self.params['image_size'])
      input_name = signitures['image_arrays'].op.name
      input_shapes = {input_name: [None, height, width, 3]}
      converter = tf.lite.TFLiteConverter.from_saved_model(
          output_dir,
          input_arrays=[input_name],
          input_shapes=input_shapes,
          output_arrays=[signitures['prediction'].op.name])
      converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
      tflite_model = converter.convert()

      tf.io.gfile.GFile(tflite_path, 'wb').write(tflite_model)
      logging.info('TFLite is saved at %s', tflite_path)

    if tensorrt:
      from tensorflow.python.compiler.tensorrt import trt  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top
      sess_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
      trt_path = os.path.join(output_dir, 'tensorrt_' + tensorrt.lower())
      trt.create_inference_graph(
          None,
          None,
          precision_mode=tensorrt,
          input_saved_model_dir=output_dir,
          output_saved_model_dir=trt_path,
          session_config=sess_config)
      logging.info('TensorRT model is saved at %s', trt_path)


class InferenceDriver(object):
  """A driver for doing batch inference.

  Example usage:

   driver = inference.InferenceDriver('efficientdet-d0', '/tmp/efficientdet-d0')
   driver.inference('/tmp/*.jpg', '/tmp/outputdir')

  """

  def __init__(self,
               model_name: Text,
               ckpt_path: Text,
               model_params: Dict[Text, Any] = None):
    """Initialize the inference driver.

    Args:
      model_name: target model name, such as efficientdet-d0.
      ckpt_path: checkpoint path, such as /tmp/efficientdet-d0/.
      model_params: model parameters for overriding the config.
    """
    self.model_name = model_name
    self.ckpt_path = ckpt_path
    self.params = hparams_config.get_detection_config(model_name).as_dict()
    if model_params:
      self.params.update(model_params)
    self.params.update(dict(is_training_bn=False))
    self.label_map = self.params.get('label_map', None)

  def inference(self, image_path_pattern: Text, output_dir: Text, **kwargs):
    """Read and preprocess input images.

    Args:
      image_path_pattern: Image file pattern such as /tmp/img*.jpg
      output_dir: the directory for output images. Output images will be named
        as 0.jpg, 1.jpg, ....
      **kwargs: extra parameters for for vistualization, such as
        min_score_thresh, max_boxes_to_draw, and line_thickness.

    Returns:
      Annotated image.
    """
    params = copy.deepcopy(self.params)
    with tf.Session() as sess:
      # Buid inputs and preprocessing.
      raw_images, images, scales = build_inputs(image_path_pattern,
                                                params['image_size'],
                                                params['mean_rgb'],
                                                params['stddev_rgb'])
      if params['data_format'] == 'channels_first':
        images = tf.transpose(images, [0, 3, 1, 2])
      # Build model.
      class_outputs, box_outputs = build_model(self.model_name, images,
                                               **self.params)
      restore_ckpt(
          sess,
          self.ckpt_path,
          ema_decay=self.params['moving_average_decay'],
          export_ckpt=None)
      # Build postprocessing.
      detections_batch = det_post_process(params, class_outputs, box_outputs,
                                          scales)
      predictions = sess.run(detections_batch)
      # Visualize results.
      for i, prediction in enumerate(predictions):
        img = visualize_image_prediction(
            raw_images[i],
            prediction,
            label_map=self.label_map,
            **kwargs)
        output_image_path = os.path.join(output_dir, str(i) + '.jpg')
        Image.fromarray(img).save(output_image_path)
        print('writing file to %s' % output_image_path)

      return predictions
