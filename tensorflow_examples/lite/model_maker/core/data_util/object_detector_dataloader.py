# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Dataloader for object detection."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import json
import os
import tempfile

from lxml import etree
import tensorflow as tf
from tensorflow_examples.lite.model_maker.core.data_util import dataloader
import yaml

from tensorflow_examples.lite.model_maker.third_party.efficientdet import dataloader as det_dataloader
from tensorflow_examples.lite.model_maker.third_party.efficientdet.dataset import create_pascal_tfrecord
from tensorflow_examples.lite.model_maker.third_party.efficientdet.dataset import tfrecord_util
from tensorflow_examples.lite.model_maker.third_party.efficientdet.keras import label_util

ANN_JSON_FILE_SUFFIX = '_annotations.json'
META_DATA_FILE_SUFFIX = '_meta_data.yaml'


def _get_cache_prefix_filename(image_dir, annotations_dir, annotations_list,
                               num_shards):
  """Get the prefix for cached files."""

  def _get_dir_basename(dirname):
    return os.path.basename(os.path.abspath(dirname))

  hasher = hashlib.md5()
  hasher.update(_get_dir_basename(image_dir).encode('utf-8'))
  hasher.update(_get_dir_basename(annotations_dir).encode('utf-8'))
  if annotations_list:
    hasher.update(' '.join(sorted(annotations_list)).encode('utf-8'))
  hasher.update(str(num_shards).encode('utf-8'))
  return hasher.hexdigest()


def _get_object_detector_cache_filenames(cache_dir,
                                         image_dir,
                                         annotations_dir,
                                         annotations_list,
                                         num_shards,
                                         cache_prefix_filename=None):
  """Gets cache filenames for obejct detector."""
  if cache_dir is None:
    cache_dir = tempfile.mkdtemp()
  if not tf.io.gfile.exists(cache_dir):
    tf.io.gfile.makedirs(cache_dir)

  if cache_prefix_filename is None:
    cache_prefix_filename = _get_cache_prefix_filename(image_dir,
                                                       annotations_dir,
                                                       annotations_list,
                                                       num_shards)
  cache_prefix = os.path.join(cache_dir, cache_prefix_filename)
  print(
      'Cache will be stored in %s with prefix filename %s. Cache_prefix is %s' %
      (cache_dir, cache_prefix_filename, cache_prefix))

  tfrecord_files = [
      cache_prefix + '-%05d-of-%05d.tfrecord' % (i, num_shards)
      for i in range(num_shards)
  ]
  annotations_json_file = cache_prefix + ANN_JSON_FILE_SUFFIX
  meta_data_file = cache_prefix + META_DATA_FILE_SUFFIX

  all_cached_files = tfrecord_files + [annotations_json_file, meta_data_file]
  is_cached = all(os.path.exists(path) for path in all_cached_files)
  return is_cached, cache_prefix, tfrecord_files, annotations_json_file, meta_data_file


def _get_label_map(label_map):
  """Gets the label map dict."""
  if isinstance(label_map, list):
    label_map_dict = {}
    for i, label in enumerate(label_map):
      # 0 is resevered for background.
      label_map_dict[i + 1] = label
    label_map = label_map_dict
  label_map = label_util.get_label_map(label_map)

  if 0 in label_map and label_map[0] != 'background':
    raise ValueError('0 must be resevered for background.')
  label_map.pop(0, None)
  name_set = set()
  for idx, name in label_map.items():
    if not isinstance(idx, int):
      raise ValueError('The key (label id) in label_map must be integer.')
    if not isinstance(name, str):
      raise ValueError('The value (label name) in label_map must be string.')
    if name in name_set:
      raise ValueError('The value: %s (label name) can\'t be duplicated.' %
                       name)
    name_set.add(name)
  return label_map


class DataLoader(dataloader.DataLoader):
  """DataLoader for object detector."""

  def __init__(self,
               tfrecord_file_patten,
               size,
               label_map,
               annotations_json_file=None):
    """Initialize DataLoader for object detector.

    Args:
      tfrecord_file_patten: Glob for tfrecord files. e.g. "/tmp/coco*.tfrecord".
      size: The size of the dataset.
      label_map: Variable shows mapping label integers ids to string label
        names. 0 is the reserved key for `background` and doesn't need to be
        included in label_map. Label names can't be duplicated. Supported
        formats are:
        1. Dict, map label integers ids to string label names, such as {1:
          'person', 2: 'notperson'}. 2. List, a list of label names such as
            ['person', 'notperson'] which is
           the same as setting label_map={1: 'person', 2: 'notperson'}.
        3. String, name for certain dataset. Accepted values are: 'coco', 'voc'
          and 'waymo'. 4. String, yaml filename that stores label_map.
      annotations_json_file: JSON with COCO data format containing golden
        bounding boxes. Used for validation. If None, use the ground truth from
        the dataloader. Refer to
        https://towardsdatascience.com/coco-data-format-for-object-detection-a4c5eaf518c5
          for the description of COCO data format.
    """
    super(DataLoader, self).__init__(dataset=None, size=size)
    self.tfrecord_file_patten = tfrecord_file_patten
    self.label_map = _get_label_map(label_map)
    self.annotations_json_file = annotations_json_file

  @classmethod
  def from_pascal_voc(cls,
                      images_dir,
                      annotations_dir,
                      label_map,
                      annotations_list=None,
                      ignore_difficult_instances=False,
                      num_shards=100,
                      max_num_images=None,
                      cache_dir=None,
                      cache_prefix_filename=None):
    """c dataset with PASCAL VOC format.

    Refer to
    https://towardsdatascience.com/coco-data-format-for-object-detection-a4c5eaf518c5#:~:text=Pascal%20VOC%20is%20an%20XML,for%20training%2C%20testing%20and%20validation
    for the description of PASCAL VOC data format.

    LabelImg Tool (https://github.com/tzutalin/labelImg) can annotate the image
    and save annotations as XML files in PASCAL VOC data format.

    Annotations are in the folder: ${annotations_dir}.
    Raw images are in the foloder: ${images_dir}.

    Args:
      images_dir: Path to directory that store raw images.
      annotations_dir: Path to the annotations directory.
      label_map: Variable shows mapping label integers ids to string label
        names. 0 is the reserved key for `background`. Label names can't be
        duplicated. Supported format: 1. Dict, map label integers ids to string
          label names, e.g.
           {1: 'person', 2: 'notperson'}. 2. List, a list of label names. e.g.
             ['person', 'notperson'] which is
           the same as setting label_map={1: 'person', 2: 'notperson'}.
        3. String, name for certain dataset. Accepted values are: 'coco', 'voc'
          and 'waymo'. 4. String, yaml filename that stores label_map.
      annotations_list: list of annotation filenames (strings) to be loaded. For
        instance, if there're 3 annotation files [0.xml, 1.xml, 2.xml] in
        `annotations_dir`, setting annotations_list=['0', '1'] makes this method
        only load [0.xml, 1.xml].
      ignore_difficult_instances: Whether to ignore difficult instances.
        `difficult` can be set inside `object` item in the annotation xml file.
      num_shards: Number of shards for output file.
      max_num_images: Max number of imags to process.
      cache_dir: The cache directory to save TFRecord, metadata and json file.
        When cache_dir is not set, a temporary folder will be created and will
        not be removed automatically after training which makes it can be used
        later.
      cache_prefix_filename: The cache prefix filename. If not set, will
        automatically generate it based on `image_dir`, `annotations_dir` and
        `annotations_list`.

    Returns:
      ObjectDetectorDataLoader object.
    """
    label_map = _get_label_map(label_map)
    is_cached, cache_prefix, tfrecord_files, ann_json_file, meta_data_file = \
        _get_object_detector_cache_filenames(cache_dir, images_dir,
                                             annotations_dir, annotations_list,
                                             num_shards, cache_prefix_filename)
    # If not cached, write data into tfrecord_file_paths and
    # annotations_json_file_path.
    # If `num_shards` differs, it's still not cached.
    if not is_cached:
      cls._write_pascal_tfrecord(images_dir, annotations_dir, label_map,
                                 annotations_list, ignore_difficult_instances,
                                 num_shards, max_num_images, tfrecord_files,
                                 ann_json_file, meta_data_file)

    return cls.from_cache(cache_prefix)

  @classmethod
  def from_cache(cls, cache_prefix):
    """Loads the data from cache.

    Args:
      cache_prefix: The cache prefix including the cache directory and the cache
        prefix filename, e.g: '/tmp/cache/train'.

    Returns:
      ObjectDetectorDataLoader object.
    """
    # Gets TFRecord files.
    tfrecord_file_patten = cache_prefix + '*.tfrecord'
    if not tf.io.gfile.glob(tfrecord_file_patten):
      raise ValueError('TFRecord files are empty.')

    # Loads meta_data.
    meta_data_file = cache_prefix + META_DATA_FILE_SUFFIX
    if not tf.io.gfile.exists(meta_data_file):
      raise ValueError('Metadata file %s doesn\'t exist.' % meta_data_file)
    with tf.io.gfile.GFile(meta_data_file, 'r') as f:
      meta_data = yaml.load(f, Loader=yaml.FullLoader)

    # Gets annotation json file.
    ann_json_file = cache_prefix + ANN_JSON_FILE_SUFFIX
    if not tf.io.gfile.exists(ann_json_file):
      ann_json_file = None

    return DataLoader(tfrecord_file_patten, meta_data['size'],
                      meta_data['label_map'], ann_json_file)

  @classmethod
  def _write_pascal_tfrecord(cls, images_dir, annotations_dir, label_map_dict,
                             annotations_list, ignore_difficult_instances,
                             num_shards, max_num_images, tfrecord_files,
                             annotations_json_file, meta_data_file):
    """Write TFRecord and json file for PASCAL VOC data."""
    label_name2id_dict = {'background': 0}
    for idx, name in label_map_dict.items():
      label_name2id_dict[name] = idx
    writers = [tf.io.TFRecordWriter(path) for path in tfrecord_files]

    ann_json_dict = {'images': [], 'annotations': [], 'categories': []}
    for class_id, class_name in label_map_dict.items():
      c = {'supercategory': 'none', 'id': class_id, 'name': class_name}
      ann_json_dict['categories'].append(c)

    # Gets the paths to annotations.
    if annotations_list:
      ann_path_list = [
          os.path.join(annotations_dir, annotation + '.xml')
          for annotation in annotations_list
      ]
    else:
      ann_path_list = list(tf.io.gfile.glob(annotations_dir + r'/*.xml'))

    for idx, ann_path in enumerate(ann_path_list):
      if max_num_images and idx >= max_num_images:
        break
      if idx % 100 == 0:
        tf.compat.v1.logging.info('On image %d of %d', idx, len(ann_path_list))
      with tf.io.gfile.GFile(ann_path, 'r') as fid:
        xml_str = fid.read()
      xml = etree.fromstring(xml_str)
      data = tfrecord_util.recursive_parse_xml_to_dict(xml)['annotation']
      tf_example = create_pascal_tfrecord.dict_to_tf_example(
          data,
          images_dir,
          label_name2id_dict,
          ignore_difficult_instances,
          ann_json_dict=ann_json_dict)
      writers[idx % num_shards].write(tf_example.SerializeToString())

    meta_data = {'size': idx + 1, 'label_map': label_map_dict}
    with tf.io.gfile.GFile(meta_data_file, 'w') as f:
      yaml.dump(meta_data, f)

    for writer in writers:
      writer.close()

    with tf.io.gfile.GFile(annotations_json_file, 'w') as f:
      json.dump(ann_json_dict, f, indent=2)

  def gen_dataset(self,
                  model_spec,
                  batch_size=None,
                  is_training=True,
                  use_fake_data=False):
    """Generate a batched tf.data.Dataset for training/evaluation.

    Args:
      model_spec: Specification for the model.
      batch_size: A integer, the returned dataset will be batched by this size.
      is_training: A boolean, when True, the returned dataset will be optionally
        shuffled and repeated as an endless dataset.
      use_fake_data: Use fake input.

    Returns:
      A TF dataset ready to be consumed by Keras model.
    """
    reader = det_dataloader.InputReader(
        self.tfrecord_file_patten,
        is_training=is_training,
        use_fake_data=use_fake_data,
        max_instances_per_image=model_spec.config.max_instances_per_image,
        debug=model_spec.config.debug)
    self._dataset = reader(model_spec.config.as_dict(), batch_size=batch_size)
    return self._dataset
