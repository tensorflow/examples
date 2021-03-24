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
"""Utilities for Object Detection Dataloader."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import io
import json
import os
from typing import Any, Dict, List, Optional

from lxml import etree
from PIL import JpegImagePlugin
import PIL.Image

import tensorflow as tf
import yaml

from tensorflow_examples.lite.model_maker.third_party.efficientdet.dataset import create_pascal_tfrecord
from tensorflow_examples.lite.model_maker.third_party.efficientdet.dataset import tfrecord_util

# A workaround to avoid a JPEG image being identified as MPO.
JpegImagePlugin._getmp = lambda: None  # pylint: disable=protected-access


class CacheFilesWriter(abc.ABC):
  """CacheFilesWriter class to write the cached files."""

  def __init__(self,
               label_map: Dict[int, str],
               images_dir: Optional[str],
               num_shards: int = 10,
               max_num_images: Optional[int] = None,
               ignore_difficult_instances: bool = False) -> None:
    """Initializes CacheFilesWriter for object detector.

    Args:
      label_map: Dict, map label integer ids to string label names such as {1:
        'person', 2: 'notperson'}. 0 is the reserved key for `background` and
          doesn't need to be included in `label_map`. Label names can't be
          duplicated.
      images_dir: Path to directory that store raw images. If None, the image
        path is the path to Google Cloud Storage or the absolute path in the
        local machine.
      num_shards: Number of shards for the output file.
      max_num_images: Max number of images to process. If None, process all the
        images.
      ignore_difficult_instances: Whether to ignore difficult instances.
        `difficult` can be set inside `object` item in the annotation xml file.
    """
    self.label_map = label_map
    self.images_dir = images_dir if images_dir else ''
    self.num_shards = num_shards
    self.max_num_images = max_num_images
    self.ignore_difficult_instances = ignore_difficult_instances
    self.unique_id = create_pascal_tfrecord.UniqueId()

    self.label_name2id_dict = {'background': 0}
    for idx, name in self.label_map.items():
      self.label_name2id_dict[name] = idx

  def write_files(self, tfrecord_files: List[str], annotations_json_file: str,
                  meta_data_file: str, *args) -> None:
    """Writes TFRecord, meta_data and annotations json files.

    Args:
      tfrecord_files: List of tfrecord files.
      annotations_json_file: Json file with COCO data format containing golden
        bounding boxes.
      meta_data_file: Yaml file to save the meta_data including data size and
        label_map.
      *args: Parameters used in the `get_tf_example` method.
    """
    writers = [tf.io.TFRecordWriter(path) for path in tfrecord_files]

    ann_json_dict = {'images': [], 'annotations': [], 'categories': []}
    for class_id, class_name in self.label_map.items():
      c = {'supercategory': 'none', 'id': class_id, 'name': class_name}
      ann_json_dict['categories'].append(c)

    # Writes tf.Example into TFRecord files.
    size = 0
    for idx, xml_dict in enumerate(self._get_xml_dict(*args)):
      if self.max_num_images and idx >= self.max_num_images:
        break
      if idx % 100 == 0:
        tf.compat.v1.logging.info('On image %d' % idx)
      tf_example = create_pascal_tfrecord.dict_to_tf_example(
          xml_dict,
          self.images_dir,
          self.label_name2id_dict,
          self.unique_id,
          ignore_difficult_instances=self.ignore_difficult_instances,
          ann_json_dict=ann_json_dict)
      writers[idx % self.num_shards].write(tf_example.SerializeToString())
      size = idx + 1

    for writer in writers:
      writer.close()

    # Writes meta_data into meta_data_file.
    meta_data = {'size': size, 'label_map': self.label_map}
    with tf.io.gfile.GFile(meta_data_file, 'w') as f:
      yaml.dump(meta_data, f)

    # Writes ann_json_dict into annotations_json_file.
    with tf.io.gfile.GFile(annotations_json_file, 'w') as f:
      json.dump(ann_json_dict, f, indent=2)

  @abc.abstractmethod
  def _get_xml_dict(self, *args) -> tf.train.Example:
    """Gets the dict holding PASCAL XML fields one by one."""
    raise NotImplementedError


class PascalVocCacheFilesWriter(CacheFilesWriter):
  """CacheFilesWriter class to write the cached files for Pascal Voc data."""

  def _get_xml_dict(
      self,
      annotations_dir: str,
      annotations_list: Optional[List[str]] = None) -> tf.train.Example:
    """Gets the tf example one by one from data with Pascal Voc format.

    Args:
      annotations_dir: Path to the annotations directory.
      annotations_list: List of annotation filenames (strings) to be loaded. For
        instance, if there're 3 annotation files [0.xml, 1.xml, 2.xml] in
        `annotations_dir`, setting annotations_list=['0', '1'] makes this method
        only load [0.xml, 1.xml].

    Yields:
      tf.train.Example
    """
    # Gets the paths to annotations.
    if annotations_list:
      ann_path_list = [
          os.path.join(annotations_dir, annotation + '.xml')
          for annotation in annotations_list
      ]
    else:
      ann_path_list = list(tf.io.gfile.glob(annotations_dir + r'/*.xml'))

    for ann_path in ann_path_list:
      with tf.io.gfile.GFile(ann_path, 'r') as fid:
        xml_str = fid.read()
      xml = etree.fromstring(xml_str)
      xml_dict = tfrecord_util.recursive_parse_xml_to_dict(xml)['annotation']
      yield xml_dict


def _get_xml_dict_from_csv_lines(images_dir: str, image_filename: str,
                                 lines: List[List[str]]) -> Dict[str, Any]:
  """Gets dict holding PASCAL VOC XML fields from the csv lines for an image."""
  image_path = os.path.join(images_dir, image_filename)
  with tf.io.gfile.GFile(image_path, 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)
  width, height = image.size

  xml_dict = {
      'size': {
          'width': width,
          'height': height
      },
      'filename': image_filename,
      'object': [],
  }

  for line in lines:
    label = line[2].strip()
    xmin, ymin = float(line[3]) * width, float(line[4]) * height
    xmax, ymax = float(line[7]) * width, float(line[8]) * height
    obj = {
        'name': label,
        'bndbox': {
            'xmin': xmin,
            'ymin': ymin,
            'xmax': xmax,
            'ymax': ymax,
            'name': label,
        },
        'difficult': 0,
        'truncated': 0,
        'pose': 'Unspecified',
    }
    xml_dict['object'].append(obj)

  return xml_dict


class CsvCacheFilesWriter(CacheFilesWriter):
  """DataLoader utilities to write the cached files for the csv file."""

  def _get_xml_dict(self, csv_lines: List[List[str]]) -> tf.train.Example:
    """Gets the tf example one by one from data with Pascal Voc format.

    Args:
      csv_lines: Lines in the csv files.

    Yields:
      tf.train.Example
    """
    image_dict = {}
    # Groups the line by image_path.
    for line in csv_lines:
      image_filename = line[1].strip()
      if image_filename not in image_dict:
        image_dict[image_filename] = []
      image_dict[image_filename].append(line)

    for image_filename, lines in image_dict.items():
      # Converts csv_lines for a certain image to dict holding PASCAL VOC XML
      # fields.
      xml_dict = _get_xml_dict_from_csv_lines(self.images_dir, image_filename,
                                              lines)
      yield xml_dict
