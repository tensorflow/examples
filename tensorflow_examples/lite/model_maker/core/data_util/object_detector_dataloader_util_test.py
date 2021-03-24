# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import filecmp
import os

import tensorflow as tf
from tensorflow_examples.lite.model_maker.core import test_util
from tensorflow_examples.lite.model_maker.core.data_util import object_detector_dataloader_util as dataloader_util
import yaml


class CacheFilesWriterTest(tf.test.TestCase):

  def test_pascal_voc_cache_writer(self):
    images_dir, annotations_dir, label_map = test_util.create_pascal_voc(
        self.get_temp_dir())

    cache_writer = dataloader_util.PascalVocCacheFilesWriter(
        label_map, images_dir, num_shards=1)

    tfrecord_files = [os.path.join(self.get_temp_dir(), 'pascal.tfrecord')]
    ann_json_file = os.path.join(self.get_temp_dir(), 'pascal_annotations.json')
    meta_data_file = os.path.join(self.get_temp_dir(), 'pascal_meta_data.yaml')
    cache_writer.write_files(tfrecord_files, ann_json_file, meta_data_file,
                             annotations_dir)

    # Checks the TFRecord file.
    self.assertTrue(os.path.isfile(tfrecord_files[0]))
    self.assertGreater(os.path.getsize(tfrecord_files[0]), 0)

    # Checks the annotation json file.
    self.assertTrue(os.path.isfile(ann_json_file))
    self.assertGreater(os.path.getsize(ann_json_file), 0)
    expected_json_file = test_util.get_test_data_path('annotations.json')
    self.assertTrue(filecmp.cmp(ann_json_file, expected_json_file))

    # Checks the meta_data file.
    self.assertTrue(os.path.isfile(meta_data_file))
    self.assertGreater(os.path.getsize(meta_data_file), 0)
    with tf.io.gfile.GFile(meta_data_file, 'r') as f:
      meta_data_dict = yaml.load(f, Loader=yaml.FullLoader)
      self.assertEqual(meta_data_dict['size'], 1)
      self.assertEqual(meta_data_dict['label_map'], label_map)

    # Checks xml_dict from `_get_xml_dict` function.
    xml_dict = next(cache_writer._get_xml_dict(annotations_dir))
    expected_xml_dict = {
        'filename': '2012_12.jpg',
        'folder': '',
        'object': [{
            'difficult': '1',
            'bndbox': {
                'xmin': '64',
                'ymin': '64',
                'xmax': '192',
                'ymax': '192',
            },
            'name': 'person',
            'truncated': '0',
            'pose': '',
        }],
        'size': {
            'width': '256',
            'height': '256',
        }
    }
    self.assertEqual(xml_dict, expected_xml_dict)

  def test_csv_cache_writer(self):
    label_map = {1: 'Baked Goods', 2: 'Cheese', 3: 'Salad'}
    images_dir = test_util.get_test_data_path('salad_images')
    cache_writer = dataloader_util.CsvCacheFilesWriter(
        label_map, images_dir, num_shards=1)

    csv_file = test_util.get_test_data_path('salads_ml_use.csv')
    for set_name, size in [('TRAIN', 1), ('TEST', 2)]:
      with tf.io.gfile.GFile(csv_file, 'r') as f:
        lines = [line for line in csv.reader(f) if line[0].startswith(set_name)]

      tfrecord_files = [
          os.path.join(self.get_temp_dir(), set_name + '_csv.tfrecord')
      ]
      ann_json_file = os.path.join(self.get_temp_dir(),
                                   set_name + '_csv_annotations.json')
      meta_data_file = os.path.join(self.get_temp_dir(),
                                    set_name + '_csv_meta_data.yaml')
      cache_writer.write_files(tfrecord_files, ann_json_file, meta_data_file,
                               lines)

      # Checks the TFRecord file.
      self.assertTrue(os.path.isfile(tfrecord_files[0]))
      self.assertGreater(os.path.getsize(tfrecord_files[0]), 0)

      # Checks the annotation json file.
      self.assertTrue(os.path.isfile(ann_json_file))
      self.assertGreater(os.path.getsize(ann_json_file), 0)
      expected_json_file = test_util.get_test_data_path(set_name.lower() +
                                                        '_annotations.json')
      self.assertTrue(filecmp.cmp(ann_json_file, expected_json_file))

      # Checks the meta_data file.
      self.assertTrue(os.path.isfile(meta_data_file))
      self.assertGreater(os.path.getsize(meta_data_file), 0)
      with tf.io.gfile.GFile(meta_data_file, 'r') as f:
        meta_data_dict = yaml.load(f, Loader=yaml.FullLoader)
        self.assertEqual(meta_data_dict['size'], size)
        self.assertEqual(meta_data_dict['label_map'], label_map)

    # Checks xml_dict from `_get_xml_dict` function.
    xml_dict = next(cache_writer._get_xml_dict(lines))
    expected_xml_dict = {
        'filename': '279324025_3e74a32a84_o.jpg',
        'object': [{
            'name': 'Baked Goods',
            'bndbox': {
                'xmin': 9.1888,
                'ymin': 101.982,
                'xmax': 908.0176,
                'ymax': 882.8832,
                'name': 'Baked Goods',
            },
            'difficult': 0,
            'truncated': 0,
            'pose': 'Unspecified',
        }],
        'size': {
            'width': 1600,
            'height': 1200,
        }
    }
    self.assertEqual(xml_dict, expected_xml_dict)


if __name__ == '__main__':
  tf.test.main()
