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
r"""Tool to inspect a model."""
import os
import time
from typing import Text, Tuple, List

from absl import app
from absl import flags
from absl import logging

import numpy as np
from PIL import Image
import tensorflow.compat.v1 as tf

from tensorflow_examples.lite.model_maker.third_party.efficientdet import hparams_config
from tensorflow_examples.lite.model_maker.third_party.efficientdet import inference
from tensorflow_examples.lite.model_maker.third_party.efficientdet import utils
from tensorflow.python.client import timeline  # pylint: disable=g-direct-tensorflow-import

flags.DEFINE_string('model_name', 'efficientdet-d0', 'Model.')
flags.DEFINE_string('logdir', '/tmp/deff/', 'log directory.')
flags.DEFINE_string('runmode', 'dry', 'Run mode: {freeze, bm, dry}')
flags.DEFINE_string('trace_filename', None, 'Trace file name.')

flags.DEFINE_integer('threads', 0, 'Number of threads.')
flags.DEFINE_integer('bm_runs', 10, 'Number of benchmark runs.')
flags.DEFINE_string('tensorrt', None, 'TensorRT mode: {None, FP32, FP16, INT8}')
flags.DEFINE_bool('delete_logdir', True, 'Whether to delete logdir.')
flags.DEFINE_bool('freeze', False, 'Freeze graph.')
flags.DEFINE_bool('use_xla', False, 'Run with xla optimization.')
flags.DEFINE_integer('batch_size', 1, 'Batch size for inference.')

flags.DEFINE_string('ckpt_path', None, 'checkpoint dir used for eval.')
flags.DEFINE_string('export_ckpt', None, 'Path for exporting new models.')

flags.DEFINE_string(
    'hparams', '', 'Comma separated k=v pairs of hyperparameters or a module'
    ' containing attributes to use as hyperparameters.')

flags.DEFINE_string('input_image', None, 'Input image path for inference.')
flags.DEFINE_string('output_image_dir', None, 'Output dir for inference.')

# For video.
flags.DEFINE_string('input_video', None, 'Input video path for inference.')
flags.DEFINE_string('output_video', None,
                    'Output video path. If None, play it online instead.')

# For visualization.
flags.DEFINE_integer('line_thickness', None, 'Line thickness for box.')
flags.DEFINE_integer('max_boxes_to_draw', 100, 'Max number of boxes to draw.')
flags.DEFINE_float('min_score_thresh', 0.4, 'Score threshold to show box.')
flags.DEFINE_string('nms_method', 'hard', 'nms method, hard or gaussian.')

# For saved model.
flags.DEFINE_string('saved_model_dir', '/tmp/saved_model',
                    'Folder path for saved model.')
flags.DEFINE_string('tflite_path', None, 'Path for exporting tflite file.')

FLAGS = flags.FLAGS


class ModelInspector(object):
  """A simple helper class for inspecting a model."""

  def __init__(self,
               model_name: Text,
               logdir: Text,
               tensorrt: Text = False,
               use_xla: bool = False,
               ckpt_path: Text = None,
               export_ckpt: Text = None,
               saved_model_dir: Text = None,
               tflite_path: Text = None,
               batch_size: int = 1,
               hparams: Text = '',
               **kwargs):
    self.model_name = model_name
    self.logdir = logdir
    self.tensorrt = tensorrt
    self.use_xla = use_xla
    self.ckpt_path = ckpt_path
    self.export_ckpt = export_ckpt
    self.saved_model_dir = saved_model_dir
    self.tflite_path = tflite_path

    model_config = hparams_config.get_detection_config(model_name)
    model_config.override(hparams)  # Add custom overrides
    model_config.is_training_bn = False
    model_config.image_size = utils.parse_image_size(model_config.image_size)

    # If batch size is 0, then build a graph with dynamic batch size.
    self.batch_size = batch_size or None
    self.labels_shape = [batch_size, model_config.num_classes]

    # A hack to make flag consistent with nms configs.
    if kwargs.get('score_thresh', None):
      model_config.nms_configs.score_thresh = kwargs['score_thresh']
    if kwargs.get('nms_method', None):
      model_config.nms_configs.method = kwargs['nms_method']
    if kwargs.get('max_output_size', None):
      model_config.nms_configs.max_output_size = kwargs['max_output_size']

    height, width = model_config.image_size
    if model_config.data_format == 'channels_first':
      self.inputs_shape = [batch_size, 3, height, width]
    else:
      self.inputs_shape = [batch_size, height, width, 3]

    self.model_config = model_config

  def build_model(self, inputs: tf.Tensor) -> List[tf.Tensor]:
    """Build model with inputs and labels and print out model stats."""
    logging.info('start building model')
    cls_outputs, box_outputs = inference.build_model(
        self.model_name,
        inputs,
        **self.model_config)

    # Write to tfevent for tensorboard.
    train_writer = tf.summary.FileWriter(self.logdir)
    train_writer.add_graph(tf.get_default_graph())
    train_writer.flush()

    all_outputs = list(cls_outputs.values()) + list(box_outputs.values())
    return all_outputs

  def export_saved_model(self, **kwargs):
    """Export a saved model for inference."""
    tf.enable_resource_variables()
    driver = inference.ServingDriver(
        self.model_name,
        self.ckpt_path,
        batch_size=self.batch_size,
        use_xla=self.use_xla,
        model_params=self.model_config.as_dict(),
        **kwargs)
    driver.build()
    driver.export(self.saved_model_dir, self.tflite_path, self.tensorrt)

  def saved_model_inference(self, image_path_pattern, output_dir, **kwargs):
    """Perform inference for the given saved model."""
    driver = inference.ServingDriver(
        self.model_name,
        self.ckpt_path,
        batch_size=self.batch_size,
        use_xla=self.use_xla,
        model_params=self.model_config.as_dict(),
        **kwargs)
    driver.load(self.saved_model_dir)

    # Serving time batch size should be fixed.
    batch_size = self.batch_size or 1
    all_files = list(tf.io.gfile.glob(image_path_pattern))
    print('all_files=', all_files)
    num_batches = (len(all_files) + batch_size - 1) // batch_size

    for i in range(num_batches):
      batch_files = all_files[i * batch_size:(i + 1) * batch_size]
      height, width = self.model_config.image_size
      images = [Image.open(f) for f in batch_files]
      if len(set([m.size for m in images])) > 1:
        # Resize only if images in the same batch have different sizes.
        images = [m.resize(height, width) for m in images]
      raw_images = [np.array(m) for m in images]
      size_before_pad = len(raw_images)
      if size_before_pad < batch_size:
        padding_size = batch_size - size_before_pad
        raw_images += [np.zeros_like(raw_images[0])] * padding_size

      detections_bs = driver.serve_images(raw_images)
      for j in range(size_before_pad):
        img = driver.visualize(raw_images[j], detections_bs[j], **kwargs)
        img_id = str(i * batch_size + j)
        output_image_path = os.path.join(output_dir, img_id + '.jpg')
        Image.fromarray(img).save(output_image_path)
        print('writing file to %s' % output_image_path)

  def saved_model_benchmark(self,
                            image_path_pattern,
                            trace_filename=None,
                            **kwargs):
    """Perform inference for the given saved model."""
    driver = inference.ServingDriver(
        self.model_name,
        self.ckpt_path,
        batch_size=self.batch_size,
        use_xla=self.use_xla,
        model_params=self.model_config.as_dict(),
        **kwargs)
    driver.load(self.saved_model_dir)
    raw_images = []
    all_files = list(tf.io.gfile.glob(image_path_pattern))
    if len(all_files) < self.batch_size:
      all_files = all_files * (self.batch_size // len(all_files) + 1)
    raw_images = [np.array(Image.open(f)) for f in all_files[:self.batch_size]]
    driver.benchmark(raw_images, trace_filename)

  def saved_model_video(self, video_path: Text, output_video: Text, **kwargs):
    """Perform video inference for the given saved model."""
    import cv2  # pylint: disable=g-import-not-at-top

    driver = inference.ServingDriver(
        self.model_name,
        self.ckpt_path,
        batch_size=1,
        use_xla=self.use_xla,
        model_params=self.model_config.as_dict())
    driver.load(self.saved_model_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
      print('Error opening input video: {}'.format(video_path))

    out_ptr = None
    if output_video:
      frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
      out_ptr = cv2.VideoWriter(output_video,
                                cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 25,
                                (frame_width, frame_height))

    while cap.isOpened():
      # Capture frame-by-frame
      ret, frame = cap.read()
      if not ret:
        break

      raw_frames = [np.array(frame)]
      detections_bs = driver.serve_images(raw_frames)
      new_frame = driver.visualize(raw_frames[0], detections_bs[0], **kwargs)

      if out_ptr:
        # write frame into output file.
        out_ptr.write(new_frame)
      else:
        # show the frame online, mainly used for real-time speed test.
        cv2.imshow('Frame', new_frame)
        # Press Q on keyboard to  exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
          break

  def inference_single_image(self, image_image_path, output_dir, **kwargs):
    driver = inference.InferenceDriver(self.model_name, self.ckpt_path,
                                       self.model_config.as_dict())
    driver.inference(image_image_path, output_dir, **kwargs)

  def build_and_save_model(self):
    """build and save the model into self.logdir."""
    with tf.Graph().as_default(), tf.Session() as sess:
      # Build model with inputs and labels.
      inputs = tf.placeholder(tf.float32, name='input', shape=self.inputs_shape)
      outputs = self.build_model(inputs)

      # Run the model
      inputs_val = np.random.rand(*self.inputs_shape).astype(float)
      labels_val = np.zeros(self.labels_shape).astype(np.int64)
      labels_val[:, 0] = 1

      if self.ckpt_path:
        # Load the true weights if available.
        inference.restore_ckpt(sess, self.ckpt_path,
                               self.model_config.moving_average_decay,
                               self.export_ckpt)
      else:
        sess.run(tf.global_variables_initializer())
        # Run a single train step.
        sess.run(outputs, feed_dict={inputs: inputs_val})

      all_saver = tf.train.Saver(save_relative_paths=True)
      all_saver.save(sess, os.path.join(self.logdir, self.model_name))

      tf_graph = os.path.join(self.logdir, self.model_name + '_train.pb')
      with tf.io.gfile.GFile(tf_graph, 'wb') as f:
        f.write(sess.graph_def.SerializeToString())

  def eval_ckpt(self):
    """build and save the model into self.logdir."""
    with tf.Graph().as_default(), tf.Session() as sess:
      # Build model with inputs and labels.
      inputs = tf.placeholder(tf.float32, name='input', shape=self.inputs_shape)
      self.build_model(inputs)
      inference.restore_ckpt(sess, self.ckpt_path,
                             self.model_config.moving_average_decay,
                             self.export_ckpt)

  def freeze_model(self) -> Tuple[Text, Text]:
    """Freeze model and convert them into tflite and tf graph."""
    with tf.Graph().as_default(), tf.Session() as sess:
      inputs = tf.placeholder(tf.float32, name='input', shape=self.inputs_shape)
      outputs = self.build_model(inputs)

      if self.ckpt_path:
        # Load the true weights if available.
        inference.restore_ckpt(sess, self.ckpt_path,
                               self.model_config.moving_average_decay,
                               self.export_ckpt)
      else:
        # Load random weights if not checkpoint is not available.
        self.build_and_save_model()
        checkpoint = tf.train.latest_checkpoint(self.logdir)
        logging.info('Loading checkpoint: %s', checkpoint)
        saver = tf.train.Saver()
        saver.restore(sess, checkpoint)

      # export frozen graph.
      output_node_names = [node.op.name for node in outputs]
      graphdef = tf.graph_util.convert_variables_to_constants(
          sess, sess.graph_def, output_node_names)

      tf_graph = os.path.join(self.logdir, self.model_name + '_frozen.pb')
      tf.io.gfile.GFile(tf_graph, 'wb').write(graphdef.SerializeToString())

      # export savaed model.
      output_dict = {'class_predict_%d' % i: outputs[i] for i in range(5)}
      output_dict.update({'box_predict_%d' % i: outputs[5+i] for i in range(5)})
      signature_def_map = {
          'serving_default':
              tf.saved_model.predict_signature_def(
                  {'input': inputs},
                  output_dict,
              )
      }
      output_dir = os.path.join(self.logdir, 'savedmodel')
      b = tf.saved_model.Builder(output_dir)
      b.add_meta_graph_and_variables(
          sess,
          tags=['serve'],
          signature_def_map=signature_def_map,
          assets_collection=tf.get_collection(tf.GraphKeys.ASSET_FILEPATHS),
          clear_devices=True)
      b.save()
      logging.info('Model saved at %s', output_dir)

    return graphdef

  def benchmark_model(self,
                      warmup_runs,
                      bm_runs,
                      num_threads,
                      trace_filename=None):
    """Benchmark model."""
    if self.tensorrt:
      print('Using tensorrt ', self.tensorrt)
      graphdef = self.freeze_model()

    if num_threads > 0:
      print('num_threads for benchmarking: {}'.format(num_threads))
      sess_config = tf.ConfigProto(
          intra_op_parallelism_threads=num_threads,
          inter_op_parallelism_threads=1)
    else:
      sess_config = tf.ConfigProto()

    sess_config.graph_options.rewrite_options.dependency_optimization = 2
    if self.use_xla:
      sess_config.graph_options.optimizer_options.global_jit_level = (
          tf.OptimizerOptions.ON_2)

    with tf.Graph().as_default(), tf.Session(config=sess_config) as sess:
      inputs = tf.placeholder(tf.float32, name='input', shape=self.inputs_shape)
      output = self.build_model(inputs)

      img = np.random.uniform(size=self.inputs_shape)

      sess.run(tf.global_variables_initializer())
      if self.tensorrt:
        fetches = [inputs.name] + [i.name for i in output]
        goutput = self.convert_tr(graphdef, fetches)
        inputs, output = goutput[0], goutput[1:]

      if not self.use_xla:
        # Don't use tf.group because XLA removes the whole graph for tf.group.
        output = tf.group(*output)
      else:
        output = tf.add_n([tf.reduce_sum(x) for x in output])

      output_name = [output.name]
      input_name = inputs.name
      graphdef = tf.graph_util.convert_variables_to_constants(
          sess, sess.graph_def, output_name)

    with tf.Graph().as_default(), tf.Session(config=sess_config) as sess:
      tf.import_graph_def(graphdef, name='')

      for i in range(warmup_runs):
        start_time = time.time()
        sess.run(output_name, feed_dict={input_name: img})
        logging.info('Warm up: {} {:.4f}s'.format(i, time.time() - start_time))

      print('Start benchmark runs total={}'.format(bm_runs))
      start = time.perf_counter()
      for i in range(bm_runs):
        sess.run(output_name, feed_dict={input_name: img})
      end = time.perf_counter()
      inference_time = (end - start) / bm_runs
      print('Per batch inference time: ', inference_time)
      print('FPS: ', self.batch_size / inference_time)

      if trace_filename:
        run_options = tf.RunOptions()
        run_options.trace_level = tf.RunOptions.FULL_TRACE
        run_metadata = tf.RunMetadata()
        sess.run(
            output_name,
            feed_dict={input_name: img},
            options=run_options,
            run_metadata=run_metadata)
        logging.info('Dumping trace to %s', trace_filename)
        trace_dir = os.path.dirname(trace_filename)
        if not tf.io.gfile.exists(trace_dir):
          tf.io.gfile.makedirs(trace_dir)
        with tf.io.gfile.GFile(trace_filename, 'w') as trace_file:
          trace = timeline.Timeline(step_stats=run_metadata.step_stats)
          trace_file.write(trace.generate_chrome_trace_format(show_memory=True))

  def convert_tr(self, graph_def, fetches):
    """Convert to TensorRT."""
    from tensorflow.python.compiler.tensorrt import trt  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top
    converter = trt.TrtGraphConverter(
        nodes_blacklist=[t.split(':')[0] for t in fetches],
        input_graph_def=graph_def,
        precision_mode=self.tensorrt)
    infer_graph = converter.convert()
    goutput = tf.import_graph_def(infer_graph, return_elements=fetches)
    return goutput

  def run_model(self, runmode, **kwargs):
    """Run the model on devices."""
    if runmode == 'dry':
      self.build_and_save_model()
    elif runmode == 'freeze':
      self.freeze_model()
    elif runmode == 'ckpt':
      self.eval_ckpt()
    elif runmode == 'saved_model_benchmark':
      self.saved_model_benchmark(
          kwargs['input_image'],
          trace_filename=kwargs.get('trace_filename', None))
    elif runmode in ('infer', 'saved_model', 'saved_model_infer',
                     'saved_model_video'):
      config_dict = {}
      if kwargs.get('line_thickness', None):
        config_dict['line_thickness'] = kwargs.get('line_thickness')
      if kwargs.get('max_boxes_to_draw', None):
        config_dict['max_boxes_to_draw'] = kwargs.get('max_boxes_to_draw')
      if kwargs.get('min_score_thresh', None):
        config_dict['min_score_thresh'] = kwargs.get('min_score_thresh')

      if runmode == 'saved_model':
        self.export_saved_model(**config_dict)
      elif runmode == 'infer':
        self.inference_single_image(kwargs['input_image'],
                                    kwargs['output_image_dir'], **config_dict)
      elif runmode == 'saved_model_infer':
        self.saved_model_inference(kwargs['input_image'],
                                   kwargs['output_image_dir'], **config_dict)
      elif runmode == 'saved_model_video':
        self.saved_model_video(kwargs['input_video'], kwargs['output_video'],
                               **config_dict)
    elif runmode == 'bm':
      self.benchmark_model(
          warmup_runs=5,
          bm_runs=kwargs.get('bm_runs', 10),
          num_threads=kwargs.get('threads', 0),
          trace_filename=kwargs.get('trace_filename', None))
    else:
      raise ValueError('Unkown runmode {}'.format(runmode))


def main(_):
  if tf.io.gfile.exists(FLAGS.logdir) and FLAGS.delete_logdir:
    logging.info('Deleting log dir ...')
    tf.io.gfile.rmtree(FLAGS.logdir)

  inspector = ModelInspector(
      model_name=FLAGS.model_name,
      logdir=FLAGS.logdir,
      tensorrt=FLAGS.tensorrt,
      use_xla=FLAGS.use_xla,
      ckpt_path=FLAGS.ckpt_path,
      export_ckpt=FLAGS.export_ckpt,
      saved_model_dir=FLAGS.saved_model_dir,
      tflite_path=FLAGS.tflite_path,
      batch_size=FLAGS.batch_size,
      hparams=FLAGS.hparams,
      score_thresh=FLAGS.min_score_thresh,
      max_output_size=FLAGS.max_boxes_to_draw,
      nms_method=FLAGS.nms_method)
  inspector.run_model(
      FLAGS.runmode,
      input_image=FLAGS.input_image,
      output_image_dir=FLAGS.output_image_dir,
      input_video=FLAGS.input_video,
      output_video=FLAGS.output_video,
      line_thickness=FLAGS.line_thickness,
      max_boxes_to_draw=FLAGS.max_boxes_to_draw,
      min_score_thresh=FLAGS.min_score_thresh,
      nms_method=FLAGS.nms_method,
      bm_runs=FLAGS.bm_runs,
      threads=FLAGS.threads,
      trace_filename=FLAGS.trace_filename)


if __name__ == '__main__':
  logging.set_verbosity(logging.WARNING)
  tf.enable_v2_tensorshape()
  tf.disable_eager_execution()
  app.run(main)
