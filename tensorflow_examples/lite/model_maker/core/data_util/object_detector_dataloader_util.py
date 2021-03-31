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

import abc
import hashlib
import io
import json
import os
import tempfile
from typing import Any, Collection, Dict, List, Sequence, Optional

import dataclasses
from lxml import etree
from PIL import JpegImagePlugin
import PIL.Image

import tensorflow as tf
import yaml

from tensorflow_examples.lite.model_maker.third_party.efficientdet.dataset import create_pascal_tfrecord
from tensorflow_examples.lite.model_maker.third_party.efficientdet.dataset import tfrecord_util

# A workaround to avoid a JPEG image being identified as MPO.
JpegImagePlugin._getmp = lambda: None  # pylint: disable=protected-access

# Suffix of the annotations json file name and the meta data file name.
ANN_JSON_FILE_SUFFIX = '_annotations.json'
META_DATA_FILE_SUFFIX = '_meta_data.yaml'


def _get_cache_dir_or_create(cache_dir: Optional[str]) -> str:
  """Gets the cache directory or creates it if not exists."""
  # TODO(b/183683348): Unifies with other tasks as well.
  # If `cache_dir` is None, a temporary folder will be created and will not be
  # removed automatically after training which makes it can be used later.
  if cache_dir is None:
    cache_dir = tempfile.mkdtemp()
  if not tf.io.gfile.exists(cache_dir):
    tf.io.gfile.makedirs(cache_dir)
  return cache_dir


def _get_dir_basename(dirname):
  """Gets the base name of the directory."""
  return os.path.basename(os.path.abspath(dirname))


def get_cache_prefix_filename_from_pascal(images_dir: str,
                                          annotations_dir: str,
                                          annotation_filenames: Optional[
                                              Collection[str]],
                                          num_shards: int = 10) -> str:
  """Gets the prefix of cached files from PASCAL VOC data.

  Args:
    images_dir: Path to directory that store raw images.
    annotations_dir: Path to the annotations directory.
    annotation_filenames: Collection of annotation filenames (strings) to be
      loaded. For instance, if there're 3 annotation files [0.xml, 1.xml, 2.xml]
      in `annotations_dir`, setting annotation_filenames=['0', '1'] makes this
      method only load [0.xml, 1.xml].
    num_shards: Number of shards for output file.

  Returns:
    The prefix of cached files.
  """
  hasher = hashlib.md5()
  hasher.update(_get_dir_basename(images_dir).encode('utf-8'))
  hasher.update(_get_dir_basename(annotations_dir).encode('utf-8'))
  if annotation_filenames:
    hasher.update(' '.join(sorted(annotation_filenames)).encode('utf-8'))
  hasher.update(str(num_shards).encode('utf-8'))
  return hasher.hexdigest()


def get_cache_prefix_filename_from_csv(csv_file: str, num_shards: int) -> str:
  """Gets the prefix of cached files from the csv file.

  Args:
    csv_file: Name of the csv file.
    num_shards: Number of shards for output file.

  Returns:
    The prefix of cached files.
  """
  hasher = hashlib.md5()
  hasher.update(os.path.basename(csv_file).encode('utf-8'))
  hasher.update(str(num_shards).encode('utf-8'))
  return hasher.hexdigest()


@dataclasses.dataclass(frozen=True)
class CacheFiles:
  """Cache files for object detection."""
  cache_prefix: str
  tfrecord_files: Sequence[str]
  meta_data_file: str
  annotations_json_file: Optional[str]


def get_cache_files(cache_dir: Optional[str],
                    cache_prefix_filename: str,
                    num_shards: int = 10) -> CacheFiles:
  """Creates an object of CacheFiles class.

  Args:
    cache_dir: The cache directory to save TFRecord, metadata and json file.
      When cache_dir is None, a temporary folder will be created and will not be
      removed automatically after training which makes it can be used later.
     cache_prefix_filename: The cache prefix filename.
     num_shards: Number of shards for output file.

  Returns:
    An object of CacheFiles class.
  """
  cache_dir = _get_cache_dir_or_create(cache_dir)
  # The cache prefix including the cache directory and the cache prefix
  # filename, e.g: '/tmp/cache/train'.
  cache_prefix = os.path.join(cache_dir, cache_prefix_filename)
  print(
      'Cache will be stored in %s with prefix filename %s. Cache_prefix is %s' %
      (cache_dir, cache_prefix_filename, cache_prefix))

  # Cached files including the TFRecord files, the annotations json file and
  # the meda data file.
  tfrecord_files = [
      cache_prefix + '-%05d-of-%05d.tfrecord' % (i, num_shards)
      for i in range(num_shards)
  ]
  annotations_json_file = cache_prefix + ANN_JSON_FILE_SUFFIX
  meta_data_file = cache_prefix + META_DATA_FILE_SUFFIX
  return CacheFiles(
      cache_prefix=cache_prefix,
      tfrecord_files=tuple(tfrecord_files),
      meta_data_file=meta_data_file,
      annotations_json_file=annotations_json_file)


def is_cached(cache_files: CacheFiles) -> bool:
  """Checks whether cache files are already cached."""
  # annotations_json_file is optional, thus we don't check whether it is cached.
  all_cached_files = list(
      cache_files.tfrecord_files) + [cache_files.meta_data_file]
  return all(tf.io.gfile.exists(path) for path in all_cached_files)


def is_all_cached(cache_files_collection: Collection[CacheFiles]) -> bool:
  """Checks whether a collection of cache files are all already cached."""
  return all(map(is_cached, cache_files_collection))


def get_cache_files_sequence(cache_dir: str, cache_prefix_filename: str,
                             set_prefixes: Collection[str],
                             num_shards: int) -> Sequence[CacheFiles]:
  """Gets a sequence of cache files.

  Args:
    cache_dir: The cache directory to save TFRecord, metadata and json file.
      When cache_dir is None, a temporary folder will be created and will not be
      removed automatically after training which makes it can be used later.
      cache_prefix_filename: The cache prefix filename.
    set_prefixes: Set prefix names for training, validation and test data. e.g.
      ['TRAIN', 'VAL', 'TEST'].
    num_shards: Number of shards for output file.

  Returns:
    A sequence of CachFiles objects mapping the set_prefixes.
  """
  cache_files_list = []
  for set_prefix in set_prefixes:
    prefix_filename = set_prefix.lower() + '_' + cache_prefix_filename
    cache_files = get_cache_files(cache_dir, prefix_filename, num_shards)
    cache_files_list.append(cache_files)
  return tuple(cache_files_list)


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

  def write_files(self, cache_files: CacheFiles, *args, **kwargs) -> None:
    """Writes TFRecord, meta_data and annotations json files.

    Args:
      cache_files: CacheFiles object including a list of TFRecord files, the
        annotations json file with COCO data format containing golden bounding
        boxes and the meda data yaml file to save the meta_data including data
        size and label_map.
      *args: Non-keyword of parameters used in the `_get_xml_dict` method.
      **kwargs: Keyword parameters used in the `_get_xml_dict` method.
    """
    writers = [
        tf.io.TFRecordWriter(path) for path in cache_files.tfrecord_files
    ]

    ann_json_dict = {'images': [], 'annotations': [], 'categories': []}
    for class_id, class_name in self.label_map.items():
      c = {'supercategory': 'none', 'id': class_id, 'name': class_name}
      ann_json_dict['categories'].append(c)

    # Writes tf.Example into TFRecord files.
    size = 0
    for idx, xml_dict in enumerate(self._get_xml_dict(*args, **kwargs)):
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
    with tf.io.gfile.GFile(cache_files.meta_data_file, 'w') as f:
      yaml.dump(meta_data, f)

    # Writes ann_json_dict into annotations_json_file.
    with tf.io.gfile.GFile(cache_files.annotations_json_file, 'w') as f:
      json.dump(ann_json_dict, f, indent=2)

  @abc.abstractmethod
  def _get_xml_dict(self, *args, **kwargs) -> tf.train.Example:
    """Gets the dict holding PASCAL XML fields one by one."""
    raise NotImplementedError


class PascalVocCacheFilesWriter(CacheFilesWriter):
  """CacheFilesWriter class to write the cached files for Pascal Voc data."""

  def _get_xml_dict(
      self,
      annotations_dir: str,
      annotation_filenames: Optional[List[str]] = None) -> tf.train.Example:
    """Gets the tf example one by one from data with Pascal Voc format.

    Args:
      annotations_dir: Path to the annotations directory.
      annotation_filenames: Collection of annotation filenames (strings) to be
        loaded. For instance, if there're 3 annotation files [0.xml, 1.xml,
        2.xml] in `annotations_dir`, setting annotation_filenames=['0', '1']
        makes this method only load [0.xml, 1.xml].

    Yields:
      tf.train.Example
    """
    # Gets the paths to annotations.
    if annotation_filenames:
      ann_path_list = [
          os.path.join(annotations_dir, annotation + '.xml')
          for annotation in annotation_filenames
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
