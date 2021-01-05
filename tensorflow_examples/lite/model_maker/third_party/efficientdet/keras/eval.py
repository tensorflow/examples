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
"""Eval libraries."""
from absl import app
from absl import flags
from absl import logging
import tensorflow as tf

from tensorflow_examples.lite.model_maker.third_party.efficientdet import coco_metric
from tensorflow_examples.lite.model_maker.third_party.efficientdet import dataloader
from tensorflow_examples.lite.model_maker.third_party.efficientdet import hparams_config
from tensorflow_examples.lite.model_maker.third_party.efficientdet import utils
from tensorflow_examples.lite.model_maker.third_party.efficientdet.keras import anchors
from tensorflow_examples.lite.model_maker.third_party.efficientdet.keras import efficientdet_keras
from tensorflow_examples.lite.model_maker.third_party.efficientdet.keras import label_util
from tensorflow_examples.lite.model_maker.third_party.efficientdet.keras import postprocess
from tensorflow_examples.lite.model_maker.third_party.efficientdet.keras import util_keras

# Cloud TPU Cluster Resolvers
flags.DEFINE_string('tpu', None, 'The Cloud TPU name.')
flags.DEFINE_string('gcp_project', None, 'Project name.')
flags.DEFINE_string('tpu_zone', None, 'GCE zone name.')

flags.DEFINE_integer('eval_samples', None, 'Number of eval samples.')
flags.DEFINE_string('val_file_pattern', None,
                    'Glob for eval tfrecords, e.g. coco/val-*.tfrecord.')
flags.DEFINE_string('val_json_file', None,
                    'Groudtruth, e.g. annotations/instances_val2017.json.')
flags.DEFINE_string('model_name', 'efficientdet-d0', 'Model name to use.')
flags.DEFINE_string('model_dir', None, 'Location of the checkpoint to run.')
flags.DEFINE_integer('batch_size', 8, 'GLobal batch size.')
flags.DEFINE_string('hparams', '', 'Comma separated k=v pairs or a yaml file')
FLAGS = flags.FLAGS


def main(_):
  config = hparams_config.get_efficientdet_config(FLAGS.model_name)
  config.override(FLAGS.hparams)
  config.val_json_file = FLAGS.val_json_file
  config.nms_configs.max_nms_inputs = anchors.MAX_DETECTION_POINTS
  config.drop_remainder = False  # eval all examples w/o drop.
  config.image_size = utils.parse_image_size(config['image_size'])

  if config.strategy == 'tpu':
    tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)
    tf.config.experimental_connect_to_cluster(tpu_cluster_resolver)
    tf.tpu.experimental.initialize_tpu_system(tpu_cluster_resolver)
    ds_strategy = tf.distribute.TPUStrategy(tpu_cluster_resolver)
    logging.info('All devices: %s', tf.config.list_logical_devices('TPU'))
  elif config.strategy == 'gpus':
    ds_strategy = tf.distribute.MirroredStrategy()
    logging.info('All devices: %s', tf.config.list_physical_devices('GPU'))
  else:
    if tf.config.list_physical_devices('GPU'):
      ds_strategy = tf.distribute.OneDeviceStrategy('device:GPU:0')
    else:
      ds_strategy = tf.distribute.OneDeviceStrategy('device:CPU:0')

  with ds_strategy.scope():
    # Network
    model = efficientdet_keras.EfficientDetNet(config=config)
    model.build((None, *config.image_size, 3))
    util_keras.restore_ckpt(model,
                            tf.train.latest_checkpoint(FLAGS.model_dir),
                            config.moving_average_decay,
                            skip_mismatch=False)
    @tf.function
    def model_fn(images, labels):
      cls_outputs, box_outputs = model(images, training=False)
      detections = postprocess.generate_detections(config,
                                                   cls_outputs,
                                                   box_outputs,
                                                   labels['image_scales'],
                                                   labels['source_ids'])
      tf.numpy_function(evaluator.update_state,
                        [labels['groundtruth_data'],
                         postprocess.transform_detections(detections)], [])

    # Evaluator for AP calculation.
    label_map = label_util.get_label_map(config.label_map)
    evaluator = coco_metric.EvaluationMetric(
        filename=config.val_json_file, label_map=label_map)

    # dataset
    batch_size = FLAGS.batch_size   # global batch size.
    ds = dataloader.InputReader(
        FLAGS.val_file_pattern,
        is_training=False,
        max_instances_per_image=config.max_instances_per_image)(
            config, batch_size=batch_size)
    if FLAGS.eval_samples:
      ds = ds.take((FLAGS.eval_samples + batch_size - 1) // batch_size)
    ds = ds_strategy.experimental_distribute_dataset(ds)

    # evaluate all images.
    eval_samples = FLAGS.eval_samples or 5000
    pbar = tf.keras.utils.Progbar((eval_samples + batch_size - 1) // batch_size)
    for i, (images, labels) in enumerate(ds):
      ds_strategy.run(model_fn, (images, labels))
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
  flags.mark_flag_as_required('val_file_pattern')
  flags.mark_flag_as_required('model_dir')
  logging.set_verbosity(logging.ERROR)
  app.run(main)
