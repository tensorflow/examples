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

from absl import app
from absl import flags
from absl import logging

import numpy as np
from PIL import Image
import tensorflow as tf

from tensorflow_examples.lite.model_maker.third_party.efficientdet import hparams_config
from tensorflow_examples.lite.model_maker.third_party.efficientdet import utils
from tensorflow_examples.lite.model_maker.third_party.efficientdet.keras import infer_lib

flags.DEFINE_string('model_name', 'efficientdet-d0', 'Model.')
flags.DEFINE_enum('mode', 'infer',
                  ['infer', 'dry', 'export', 'benchmark', 'video'], 'Run mode.')
flags.DEFINE_string('trace_filename', None, 'Trace file name.')

flags.DEFINE_integer('bm_runs', 10, 'Number of benchmark runs.')
flags.DEFINE_enum('tensorrt', '', ['', 'FP32', 'FP16', 'INT8'],
                  'TensorRT mode.')
flags.DEFINE_integer('batch_size', 1, 'Batch size for inference.')
flags.DEFINE_integer('image_size', -1, 'Input image size for inference.')

flags.DEFINE_string('model_dir', '_', 'checkpoint dir used for eval.')
flags.DEFINE_string('export_ckpt', None, 'Output model ckpt path.')

flags.DEFINE_string(
    'hparams', '', 'Comma separated k=v pairs of hyperparameters or a module'
    ' containing attributes to use as hyperparameters.')

flags.DEFINE_string('input_image', None, 'Input image path for inference.')
flags.DEFINE_string('output_image_dir', None, 'Output dir for inference.')

# For video.
flags.DEFINE_string('input_video', None, 'Input video path for inference.')
flags.DEFINE_string('output_video', None,
                    'Output video path. If None, play it online instead.')

# For saved model.
flags.DEFINE_string('saved_model_dir', None, 'Folder path for saved model.')
flags.DEFINE_enum('tflite', '', ['', 'FP32', 'FP16', 'INT8'], 'tflite type.')
flags.DEFINE_string('file_pattern', None,
                    'Glob for tfrecords, e.g. coco/val-*.tfrecord.')
flags.DEFINE_integer(
    'num_calibration_steps', 2000,
    'Number of post-training quantization calibration steps to run.')
flags.DEFINE_bool('debug', False, 'Debug mode.')
flags.DEFINE_bool('only_network', False, 'Model only contains network')
FLAGS = flags.FLAGS


def main(_):
  tf.config.run_functions_eagerly(FLAGS.debug)
  devices = tf.config.list_physical_devices('GPU')
  for device in devices:
    tf.config.experimental.set_memory_growth(device, True)

  model_config = hparams_config.get_detection_config(FLAGS.model_name)
  model_config.override(FLAGS.hparams)  # Add custom overrides
  model_config.is_training_bn = False
  if FLAGS.image_size != -1:
    model_config.image_size = FLAGS.image_size
  model_config.image_size = utils.parse_image_size(model_config.image_size)

  model_params = model_config.as_dict()
  ckpt_path_or_file = FLAGS.model_dir
  if tf.io.gfile.isdir(ckpt_path_or_file):
    ckpt_path_or_file = tf.train.latest_checkpoint(ckpt_path_or_file)
  driver = infer_lib.ServingDriver(FLAGS.model_name, ckpt_path_or_file,
                                   FLAGS.batch_size or None,
                                   FLAGS.only_network, model_params)
  if FLAGS.mode == 'export':
    if not FLAGS.saved_model_dir:
      raise ValueError('Please specify --saved_model_dir=')
    model_dir = FLAGS.saved_model_dir
    if tf.io.gfile.exists(model_dir):
      tf.io.gfile.rmtree(model_dir)
    driver.export(model_dir, FLAGS.tensorrt, FLAGS.tflite, FLAGS.file_pattern,
                  FLAGS.num_calibration_steps)
    print('Model are exported to %s' % model_dir)
  elif FLAGS.mode == 'infer':
    image_file = tf.io.read_file(FLAGS.input_image)
    image_arrays = tf.io.decode_image(image_file)
    image_arrays.set_shape((None, None, 3))
    image_arrays = tf.expand_dims(image_arrays, axis=0)
    if FLAGS.saved_model_dir:
      driver.load(FLAGS.saved_model_dir)
      if FLAGS.saved_model_dir.endswith('.tflite'):
        image_size = utils.parse_image_size(model_config.image_size)
        image_arrays = tf.image.resize_with_pad(image_arrays, *image_size)
        image_arrays = tf.cast(image_arrays, tf.uint8)
    detections_bs = driver.serve(image_arrays)
    boxes, scores, classes, _ = tf.nest.map_structure(np.array, detections_bs)
    raw_image = Image.fromarray(np.array(image_arrays)[0])
    img = driver.visualize(
        raw_image,
        boxes[0],
        classes[0],
        scores[0],
        min_score_thresh=model_config.nms_configs.score_thresh or 0.4,
        max_boxes_to_draw=model_config.nms_configs.max_output_size)
    output_image_path = os.path.join(FLAGS.output_image_dir, '0.jpg')
    Image.fromarray(img).save(output_image_path)
    print('writing file to %s' % output_image_path)
  elif FLAGS.mode == 'benchmark':
    if FLAGS.saved_model_dir:
      driver.load(FLAGS.saved_model_dir)

    batch_size = FLAGS.batch_size or 1
    if FLAGS.input_image:
      image_file = tf.io.read_file(FLAGS.input_image)
      image_arrays = tf.image.decode_image(image_file)
      image_arrays.set_shape((None, None, 3))
      image_arrays = tf.expand_dims(image_arrays, 0)
      if batch_size > 1:
        image_arrays = tf.tile(image_arrays, [batch_size, 1, 1, 1])
    else:
      # use synthetic data if no image is provided.
      image_arrays = tf.ones((batch_size, *model_config.image_size, 3),
                             dtype=tf.uint8)
    if FLAGS.only_network:
      image_arrays = tf.image.convert_image_dtype(image_arrays, tf.float32)
      image_arrays = tf.image.resize(image_arrays, model_config.image_size)
    driver.benchmark(image_arrays, FLAGS.bm_runs, FLAGS.trace_filename)
  elif FLAGS.mode == 'dry':
    # transfer to tf2 format ckpt
    driver.build()
    if FLAGS.export_ckpt:
      driver.model.save_weights(FLAGS.export_ckpt)
  elif FLAGS.mode == 'video':
    import cv2  # pylint: disable=g-import-not-at-top
    if FLAGS.saved_model_dir:
      driver.load(FLAGS.saved_model_dir)
    cap = cv2.VideoCapture(FLAGS.input_video)
    if not cap.isOpened():
      print('Error opening input video: {}'.format(FLAGS.input_video))

    out_ptr = None
    if FLAGS.output_video:
      frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
      out_ptr = cv2.VideoWriter(FLAGS.output_video,
                                cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                                cap.get(5), (frame_width, frame_height))

    while cap.isOpened():
      # Capture frame-by-frame
      ret, frame = cap.read()
      if not ret:
        break

      raw_frames = np.array([frame])
      detections_bs = driver.serve(raw_frames)
      boxes, scores, classes, _ = tf.nest.map_structure(np.array, detections_bs)
      new_frame = driver.visualize(
          raw_frames[0],
          boxes[0],
          classes[0],
          scores[0],
          min_score_thresh=model_config.nms_configs.score_thresh or 0.4,
          max_boxes_to_draw=model_config.nms_configs.max_output_size)

      if out_ptr:
        # write frame into output file.
        out_ptr.write(new_frame)
      else:
        # show the frame online, mainly used for real-time speed test.
        cv2.imshow('Frame', new_frame)
        # Press Q on keyboard to  exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
          break


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  app.run(main)
