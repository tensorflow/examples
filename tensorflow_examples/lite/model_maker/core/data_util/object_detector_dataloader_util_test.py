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

import csv
import filecmp
import os

import tensorflow as tf
from tensorflow_examples.lite.model_maker.core import test_util
from tensorflow_examples.lite.model_maker.core.data_util import object_detector_dataloader_util as dataloader_util
import yaml


class CacheFilesTest(tf.test.TestCase):

  def test_get_cache_files(self):
    cache_files = dataloader_util.get_cache_files(
        cache_dir='/tmp/', cache_prefix_filename='train', num_shards=1)
    self.assertEqual(cache_files.cache_prefix, '/tmp/train')
    self.assertLen(cache_files.tfrecord_files, 1)
    self.assertEqual(cache_files.tfrecord_files[0],
                     '/tmp/train-00000-of-00001.tfrecord')
    self.assertEqual(cache_files.meta_data_file, '/tmp/train_meta_data.yaml')
    self.assertEqual(cache_files.annotations_json_file,
                     '/tmp/train_annotations.json')

  def test_filename_from_pascal(self):
    # Checks the filenames are not equal if any of the parameters is changed.
    images_dir = '/tmp/images/'
    annotations_dir = '/tmp/annotations/'
    annotation_filenames = None
    num_shards = 1
    filename = dataloader_util.get_cache_prefix_filename_from_pascal(
        images_dir=images_dir,
        annotations_dir=annotations_dir,
        annotation_filenames=annotation_filenames,
        num_shards=num_shards)

    images_dir = '/tmp/images_1/'
    filename1 = dataloader_util.get_cache_prefix_filename_from_pascal(
        images_dir=images_dir,
        annotations_dir=annotations_dir,
        annotation_filenames=annotation_filenames,
        num_shards=num_shards)
    self.assertNotEqual(filename, filename1)

    annotations_dir = '/tmp/annotations_2/'
    filename2 = dataloader_util.get_cache_prefix_filename_from_pascal(
        images_dir=images_dir,
        annotations_dir=annotations_dir,
        annotation_filenames=annotation_filenames,
        num_shards=num_shards)
    self.assertNotEqual(filename1, filename2)

    annotation_filenames = ['1', '2']
    filename3 = dataloader_util.get_cache_prefix_filename_from_pascal(
        images_dir=images_dir,
        annotations_dir=annotations_dir,
        annotation_filenames=annotation_filenames,
        num_shards=num_shards)
    self.assertNotEqual(filename2, filename3)

    num_shards = 10
    filename4 = dataloader_util.get_cache_prefix_filename_from_pascal(
        images_dir=images_dir,
        annotations_dir=annotations_dir,
        annotation_filenames=annotation_filenames,
        num_shards=num_shards)
    self.assertNotEqual(filename3, filename4)

  def test_filename_from_csv(self):
    # Checks the filenames are not equal if any of the parameters is changed.
    csv_file = '/tmp/1.csv'
    num_shards = 1
    filename = dataloader_util.get_cache_prefix_filename_from_csv(
        csv_file, num_shards)

    csv_file = '/tmp/2.csv'
    filename1 = dataloader_util.get_cache_prefix_filename_from_csv(
        csv_file, num_shards)
    self.assertNotEqual(filename, filename1)

    num_shards = 10
    filename2 = dataloader_util.get_cache_prefix_filename_from_csv(
        csv_file, num_shards)
    self.assertNotEqual(filename1, filename2)


class CacheFilesWriterTest(tf.test.TestCase):

  def test_pascal_voc_cache_writer(self):
    images_dir, annotations_dir, label_map = test_util.create_pascal_voc(
        self.get_temp_dir())

    cache_writer = dataloader_util.PascalVocCacheFilesWriter(
        label_map, images_dir, num_shards=1)

    cache_files = dataloader_util.get_cache_files(
        cache_dir=self.get_temp_dir(), cache_prefix_filename='pascal')
    cache_writer.write_files(cache_files, annotations_dir)

    # Checks the TFRecord file.
    tfrecord_files = cache_files.tfrecord_files
    self.assertTrue(os.path.isfile(tfrecord_files[0]))
    self.assertGreater(os.path.getsize(tfrecord_files[0]), 0)

    # Checks the annotation json file.
    annotations_json_file = cache_files.annotations_json_file
    self.assertTrue(os.path.isfile(annotations_json_file))
    self.assertGreater(os.path.getsize(annotations_json_file), 0)
    expected_json_file = test_util.get_test_data_path('annotations.json')
    self.assertTrue(filecmp.cmp(annotations_json_file, expected_json_file))

    # Checks the meta_data file.
    meta_data_file = cache_files.meta_data_file
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

      cache_files = dataloader_util.get_cache_files(
          cache_dir=self.get_temp_dir(), cache_prefix_filename='csv')
      cache_writer.write_files(cache_files, lines)

      # Checks the TFRecord file.
      tfrecord_files = cache_files.tfrecord_files
      self.assertTrue(os.path.isfile(tfrecord_files[0]))
      self.assertGreater(os.path.getsize(tfrecord_files[0]), 0)

      # Checks the annotation json file.
      annotations_json_file = cache_files.annotations_json_file
      self.assertTrue(os.path.isfile(annotations_json_file))
      self.assertGreater(os.path.getsize(annotations_json_file), 0)
      expected_json_file = test_util.get_test_data_path(set_name.lower() +
                                                        '_annotations.json')
      self.assertTrue(filecmp.cmp(annotations_json_file, expected_json_file))

      # Checks the meta_data file.
      meta_data_file = cache_files.meta_data_file
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
