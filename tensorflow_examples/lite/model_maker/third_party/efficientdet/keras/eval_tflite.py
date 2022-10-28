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
"""Eval libraries. Used for TFLite model without post-processing."""
from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow as tf

from tensorflow_examples.lite.model_maker.third_party.efficientdet import coco_metric
from tensorflow_examples.lite.model_maker.third_party.efficientdet import dataloader
from tensorflow_examples.lite.model_maker.third_party.efficientdet import hparams_config
from tensorflow_examples.lite.model_maker.third_party.efficientdet import utils

from tensorflow_examples.lite.model_maker.third_party.efficientdet.keras import anchors
from tensorflow_examples.lite.model_maker.third_party.efficientdet.keras import label_util
from tensorflow_examples.lite.model_maker.third_party.efficientdet.keras import postprocess

FLAGS = flags.FLAGS

DEFAULT_SCALE, DEFAULT_ZERO_POINT = 0, 0


def define_flags():
  """Define the flags."""
  flags.DEFINE_integer('eval_samples', None, 'Number of eval samples.')
  flags.DEFINE_string('val_file_pattern', None,
                      'Glob for eval tfrecords, e.g. coco/val-*.tfrecord.')
  flags.DEFINE_string('val_json_file', None,
                      'Groudtruth, e.g. annotations/instances_val2017.json.')
  flags.DEFINE_string('model_name', 'efficientdet-d0', 'Model name to use.')
  flags.DEFINE_string('tflite_path', None, 'Path to TFLite model.')
  flags.DEFINE_bool(
      'only_network', False,
      'TFLite model only contains EfficientDetNet without post-processing NMS op.'
  )
  flags.DEFINE_bool('pre_class_nms', False, 'Use pre_class_nms for evaluation.')
  flags.DEFINE_string('hparams', '', 'Comma separated k=v pairs or a yaml file')
  flags.mark_flag_as_required('val_file_pattern')
  flags.mark_flag_as_required('tflite_path')


class LiteRunner(object):
  """Runs inference with TF Lite model."""

  def __init__(self, tflite_model_path, only_network=False):
    """Initializes Lite runner with tflite model file.

    Args:
      tflite_model_path: Path to TFLite model file.
      only_network: boolean, If True, TFLite model only contains EfficientDetNet
        without post-processing NMS op. If False, TFLite model contains custom
        NMS op.
    """
    self.interpreter = tf.lite.Interpreter(tflite_model_path)
    self.interpreter.allocate_tensors()
    # Get input and output tensors.
    self.input_details = self.interpreter.get_input_details()
    self.output_details = self.interpreter.get_output_details()
    self.only_network = only_network
    if self.only_network:
      # TFLite with only_network=True changed the order of output. Sort it
      # according to tensor name.
      # TFLite with only_network=False  doesn't change the order since the
      # latest op is cutom NMS op.
      self.output_details = sorted(self.output_details, key=lambda a: a['name'])

  def run(self, image):
    """Runs inference with Lite model."""
    interpreter = self.interpreter
    signature_fn = interpreter.get_signature_runner()
    input_details = self.input_details
    output_details = self.output_details

    input_detail = input_details[0]
    if input_detail['quantization'] != (DEFAULT_SCALE, DEFAULT_ZERO_POINT):
      scale, zero_point = input_detail['quantization']
      image = image / scale + zero_point
      image = np.array(image, dtype=input_detail['dtype'])

    output = signature_fn(images=image)

    def get_output(idx):
      output_detail = output_details[idx]
      output_tensor = output[f'output_{idx}']
      if output_detail['quantization'] != (DEFAULT_SCALE, DEFAULT_ZERO_POINT):
        # Dequantize the output
        scale, zero_point = output_detail['quantization']
        output_tensor = output_tensor.astype(np.float32)
        output_tensor = (output_tensor - zero_point) * scale
      return output_tensor

    output_size = len(output_details)
    if not self.only_network:
      # TFLite model with post-processing.
      # Four Outputs:
      #   num_boxes: a float32 tensor of size 1 containing the number of
      #     detected boxes
      #   detection_scores: a float32 tensor of shape [1, num_boxes]
      #     with class scores
      #   detection_classes: a float32 tensor of shape [1, num_boxes]
      #     with class indices
      #   detection_boxes: a float32 tensor of shape [1, num_boxes, 4] with box
      #     locations
      return [get_output(i) for i in range(output_size)]
    else:
      # TFLite model only contains network without post-processing.
      num_boxes = int(output_size / 2)
      cls_outputs, box_outputs = [], []
      for i in range(num_boxes):
        cls_outputs.append(get_output(i))
        box_outputs.append(get_output(i + num_boxes))
      return cls_outputs, box_outputs


def main(_):
  config = hparams_config.get_efficientdet_config(FLAGS.model_name)
  config.override(FLAGS.hparams)
  config.val_json_file = FLAGS.val_json_file
  config.nms_configs.max_nms_inputs = anchors.MAX_DETECTION_POINTS
  config.drop_remainder = False  # eval all examples w/o drop.
  config.image_size = utils.parse_image_size(config['image_size'])

  # Evaluator for AP calculation.
  label_map = label_util.get_label_map(config.label_map)
  evaluator = coco_metric.EvaluationMetric(
      filename=config.val_json_file, label_map=label_map)

  # dataset
  batch_size = 1
  ds = dataloader.InputReader(
      FLAGS.val_file_pattern,
      is_training=False,
      max_instances_per_image=config.max_instances_per_image)(
          config, batch_size=batch_size)
  eval_samples = FLAGS.eval_samples
  if eval_samples:
    ds = ds.take((eval_samples + batch_size - 1) // batch_size)

  # Network
  lite_runner = LiteRunner(FLAGS.tflite_path, FLAGS.only_network)
  eval_samples = FLAGS.eval_samples or 5000
  pbar = tf.keras.utils.Progbar((eval_samples + batch_size - 1) // batch_size)
  for i, (images, labels) in enumerate(ds):
    if not FLAGS.only_network:
      _, nms_scores_bs, nms_classes_bs, nms_boxes_bs = lite_runner.run(images)
      nms_classes_bs += postprocess.CLASS_OFFSET

      height, width = utils.parse_image_size(config.image_size)
      normalize_factor = tf.constant([height, width, height, width],
                                     dtype=tf.float32)
      nms_boxes_bs *= normalize_factor
      if labels['image_scales'] is not None:
        scales = tf.expand_dims(tf.expand_dims(labels['image_scales'], -1), -1)
        nms_boxes_bs = nms_boxes_bs * tf.cast(scales, nms_boxes_bs.dtype)
      detections = postprocess.generate_detections_from_nms_output(
          nms_boxes_bs, nms_classes_bs, nms_scores_bs, labels['source_ids'])
    else:
      cls_outputs, box_outputs = lite_runner.run(images)
      detections = postprocess.generate_detections(
          config,
          cls_outputs,
          box_outputs,
          labels['image_scales'],
          labels['source_ids'],
          pre_class_nms=FLAGS.pre_class_nms)

    detections = postprocess.transform_detections(detections)
    evaluator.update_state(labels['groundtruth_data'].numpy(),
                           detections.numpy())
    pbar.update(i)

  # compute the final eval results.
  metrics = evaluator.result()
  metric_dict = {}
  for i, name in enumerate(evaluator.metric_names):
    metric_dict[name] = metrics[i]

  if label_map:
    for i, cid in enumerate(sorted(label_map.keys())):
      name = 'AP_/%s' % label_map[cid]
      metric_dict[name] = metrics[i + len(evaluator.metric_names)]
  print(FLAGS.model_name, metric_dict)


if __name__ == '__main__':
  define_flags()
  logging.set_verbosity(logging.WARNING)
  app.run(main)
