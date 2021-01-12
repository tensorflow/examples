# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for object detector specs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os

import tensorflow.compat.v2 as tf

from tensorflow_examples.lite.model_maker.core import test_util
from tensorflow_examples.lite.model_maker.core.task.model_spec import object_detector_spec


class EfficientDetModelSpecTest(tf.test.TestCase):

  @classmethod
  def setUpClass(cls):
    super(EfficientDetModelSpecTest, cls).setUpClass()
    hub_path = test_util.get_test_data_path('fake_effdet_lite0_hub')
    cls._spec = object_detector_spec.EfficientDetModelSpec(
        model_name='efficientdet-lite0', uri=hub_path, hparams=dict(map_freq=1))
    cls.model = cls._spec.create_model()

  def test_create_model(self):
    self.assertIsInstance(self.model, tf.keras.Model)
    x = tf.ones((1, *self._spec.config.image_size, 3))
    cls_outputs, box_outputs = self.model(x)
    self.assertLen(cls_outputs, 5)
    self.assertLen(box_outputs, 5)

  def test_train(self):
    model = self._spec.train(
        self.model,
        train_dataset=self._gen_input(),
        steps_per_epoch=1,
        val_dataset=self._gen_input(),
        validation_steps=1,
        epochs=1,
        batch_size=1)
    self.assertIsInstance(model, tf.keras.Model)

  def test_evaluate(self):
    metrics = self._spec.evaluate(
        self.model, dataset=self._gen_input(), steps=1)
    self.assertIsInstance(metrics, dict)
    self.assertGreaterEqual(metrics['AP'], 0)

  def _gen_input(self):
    # Image tensors that are preprocessed to have normalized value and fixed
    # dimension [1, image_height, image_width, 3]
    images = tf.random.uniform((1, 320, 320, 3), maxval=256)

    # labels contains:
    # box_targets_dict: ordered dictionary with keys
    #     [min_level, min_level+1, ..., max_level]. The values are tensor with
    #     shape [height_l, width_l, num_anchors * 4]. The height_l and
    #     width_l represent the dimension of bounding box regression output at
    #     l-th level.
    # cls_targets_dict: ordered dictionary with keys
    #     [min_level, min_level+1, ..., max_level]. The values are tensor with
    #     shape [height_l, width_l, num_anchors]. The height_l and width_l
    #     represent the dimension of class logits at l-th level.
    # groundtruth_data: Groundtruth Annotations data.
    # image_scale: Scale of the processed image to the original image.
    # source_id: Source image id. Default value -1 if the source id is empty
    #     in the groundtruth annotation.
    # mean_num_positives:  Mean number of positive anchors in the batch images.
    sizes = [(level, math.ceil(320 / 2**level)) for level in range(3, 8)]

    labels = {
        'box_targets_%d' % level: tf.ones((1, size, size, 36))
        for level, size in sizes
    }
    labels.update({
        'cls_targets_%d' % level: tf.ones((1, size, size, 9), dtype=tf.int32)
        for level, size in sizes
    })
    labels.update({'groundtruth_data': tf.zeros([1, 100, 7])})
    labels.update({'image_scales': tf.constant([0.8])})
    labels.update({'source_ids': tf.constant([1.0])})
    labels.update({'mean_num_positives': tf.constant([10.0])})
    ds = tf.data.Dataset.from_tensors((images, labels))
    return ds


if __name__ == '__main__':
  # Load compressed models from tensorflow_hub
  os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'
  tf.test.main()
