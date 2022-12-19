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
"""Inspect tfrecord dataset."""
import os
from absl import app
from absl import flags
from absl import logging
import numpy as np
from PIL import Image
import tensorflow as tf

from tensorflow_examples.lite.model_maker.third_party.efficientdet import dataloader
from tensorflow_examples.lite.model_maker.third_party.efficientdet import hparams_config
from tensorflow_examples.lite.model_maker.third_party.efficientdet import utils
from tensorflow_examples.lite.model_maker.third_party.efficientdet.visualize import vis_utils

flags.DEFINE_string('save_samples_dir', 'tfrecord_samples',
                    'Location of samples to save')
flags.DEFINE_string('model_name', 'efficientdet-d0',
                    'model name for config and image_size')
flags.DEFINE_string(
    'hparams', '', 'Comma separated k=v pairs of hyperparameters or a module'
    ' containing attributes to use as hyperparameters.')
flags.DEFINE_integer('samples', 10,
                     'Number of random samples for visualization.')
flags.DEFINE_string('file_pattern', None,
                    'Glob for data files (e.g., COCO train - minival set)')
flags.DEFINE_bool('eval', False, 'flag for file pattern mode i.e eval')
FLAGS = flags.FLAGS


class RecordInspect:
  """Inspection Class."""

  def __init__(self, config):
    """Initializes RecordInspect with passed config.

    Args:
        config: config file to initialize input_fn.
    """
    self.input_fn = dataloader.InputReader(
        FLAGS.file_pattern,
        is_training=not FLAGS.eval,
        use_fake_data=False,
        max_instances_per_image=config.max_instances_per_image)

    self.params = dict(
        config.as_dict(), batch_size=FLAGS.samples, model_name=FLAGS.model_name)
    logging.info(self.params)
    self.cls_to_label = config.label_map
    os.makedirs(FLAGS.save_samples_dir, exist_ok=True)

  def visualize(self):
    """save tfrecords images with bounding boxes."""
    vis_ds = self.input_fn(params=self.params)
    data = next(iter(vis_ds))  # iterable.
    images = data[0]
    gt_data = data[1]['groundtruth_data']

    # scales
    scale_to_org = data[1]['image_scales']
    scales = 1.0 / scale_to_org
    offset = tf.constant([0.485, 0.456, 0.406])
    offset = tf.reshape(offset, (1, 1, -1))
    scale_image = tf.constant([0.229, 0.224, 0.225])
    scale_image = tf.reshape(scale_image, (1, 1, -1))

    logging.info('Visualizing TfRecords %s', FLAGS.file_pattern)
    for i, zip_data in enumerate(zip(gt_data, images, scales)):
      gt, image, scale = zip_data
      boxes = gt[:, :4]
      boxes = boxes[np.any(boxes > 0, axis=1)].numpy()
      if boxes.shape[0] > 0:
        classes = gt[:boxes.shape[0], -1].numpy()
        try:
          category_index = {idx: {'id': idx, 'name': self.cls_to_label[idx]}
                            for idx in np.asarray(classes, dtype=int)}
        except Exception:  # pylint: disable=broad-except
          category_index = {}

        # unnormalize image.
        image *= scale_image
        image += offset

        # 0-255. range
        image = np.asarray(image.numpy() * 255., dtype=np.uint8)

        # scale to image_size
        boxes *= scale.numpy()

        image = vis_utils.visualize_boxes_and_labels_on_image_array(
            image,
            boxes=boxes,
            classes=classes.astype(int),
            scores=np.ones(boxes.shape[0]),
            category_index=category_index,
            line_thickness=2,
            skip_scores=True)
        image = Image.fromarray(image)
        image.save(os.path.join(FLAGS.save_samples_dir, f'sample{i}.jpg'))


def main(_):
  # Parse and override hparams
  config = hparams_config.get_detection_config(FLAGS.model_name)
  config.override(FLAGS.hparams)

  # Parse image size in case it is in string format.
  config.image_size = utils.parse_image_size(config.image_size)
  try:
    recordinspect = RecordInspect(config)
    recordinspect.visualize()
  except Exception as e:  # pylint: disable=broad-except
    logging.error(e)
  else:
    logging.info('Done, please find samples at %s', FLAGS.save_samples_dir)


if __name__ == '__main__':
  app.run(main)
