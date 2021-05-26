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

import csv
from typing import Collection, Dict, List, Optional, Tuple, TypeVar, Union

import tensorflow as tf
from tensorflow_examples.lite.model_maker.core.api.api_util import mm_export
from tensorflow_examples.lite.model_maker.core.data_util import dataloader
from tensorflow_examples.lite.model_maker.core.data_util import object_detector_dataloader_util as util
import yaml

from tensorflow_examples.lite.model_maker.third_party.efficientdet import dataloader as det_dataloader
from tensorflow_examples.lite.model_maker.third_party.efficientdet.keras import label_util

DetectorDataLoader = TypeVar('DetectorDataLoader', bound='DataLoader')
# Csv lines with the label map.
CsvLines = Tuple[List[List[List[str]]], Dict[int, str]]


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


def _group_csv_lines(csv_file: str,
                     set_prefixes: List[str],
                     delimiter: str = ',',
                     quotechar: str = '"') -> CsvLines:
  """Groups csv_lines for different set_names and label_map.

  Args:
    csv_file: filename of the csv file.
    set_prefixes: Set prefix names for training, validation and test data. e.g.
      ['TRAIN', 'VAL', 'TEST'].
    delimiter: Character used to separate fields.
    quotechar: Character used to quote fields containing special characters.

  Returns:
    [training csv lines, validation csv lines, test csv lines], label_map
  """
  # Dict that maps integer label ids to string label names.
  label_map = {}
  with tf.io.gfile.GFile(csv_file, 'r') as f:
    reader = csv.reader(f, delimiter=delimiter, quotechar=quotechar)
    # `lines_list` = [training csv lines, validation csv lines, test csv lines]
    # Each csv line is a list of strings separated by delimiter. e.g.
    # row 'one,two,three' in the csv file will be ['one', two', 'three'].
    lines_list = [[], [], []]
    for line in reader:
      # Groups lines by the set_name.
      set_name = line[0].strip()
      for i, set_prefix in enumerate(set_prefixes):
        if set_name.startswith(set_prefix):
          lines_list[i].append(line)

      label = line[2].strip()
      # Updates label_map if it's a new label.
      if label not in label_map.values():
        label_map[len(label_map) + 1] = label

  return lines_list, label_map


@mm_export('object_detector.DataLoader')
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
  def from_pascal_voc(
      cls,
      images_dir: str,
      annotations_dir: str,
      label_map: Union[List[str], Dict[int, str], str],
      annotation_filenames: Optional[Collection[str]] = None,
      ignore_difficult_instances: bool = False,
      num_shards: int = 100,
      max_num_images: Optional[int] = None,
      cache_dir: Optional[str] = None,
      cache_prefix_filename: Optional[str] = None) -> DetectorDataLoader:
    """Loads from dataset with PASCAL VOC format.

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
      annotation_filenames: Collection of annotation filenames (strings) to be
        loaded. For instance, if there're 3 annotation files [0.xml, 1.xml,
        2.xml] in `annotations_dir`, setting annotation_filenames=['0', '1']
        makes this method only load [0.xml, 1.xml].
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
        `annotation_filenames`.

    Returns:
      ObjectDetectorDataLoader object.
    """
    label_map = _get_label_map(label_map)

    # If `cache_prefix_filename` is None, automatically generates a hash value.
    if cache_prefix_filename is None:
      cache_prefix_filename = util.get_cache_prefix_filename_from_pascal(
          images_dir=images_dir,
          annotations_dir=annotations_dir,
          annotation_filenames=annotation_filenames,
          num_shards=num_shards)

    cache_files = util.get_cache_files(
        cache_dir=cache_dir,
        cache_prefix_filename=cache_prefix_filename,
        num_shards=num_shards)

    # If not cached, writes data into tfrecord_file_paths and
    # annotations_json_file_path.
    # If `num_shards` differs, it's still not cached.
    if not util.is_cached(cache_files):
      cache_writer = util.PascalVocCacheFilesWriter(
          label_map=label_map,
          images_dir=images_dir,
          num_shards=num_shards,
          max_num_images=max_num_images,
          ignore_difficult_instances=ignore_difficult_instances)
      cache_writer.write_files(
          cache_files=cache_files,
          annotations_dir=annotations_dir,
          annotation_filenames=annotation_filenames)

    return cls.from_cache(cache_files.cache_prefix)

  @classmethod
  def from_csv(
      cls,
      filename: str,
      images_dir: Optional[str] = None,
      delimiter: str = ',',
      quotechar: str = '"',
      num_shards: int = 10,
      max_num_images: Optional[int] = None,
      cache_dir: Optional[str] = None,
      cache_prefix_filename: Optional[str] = None
  ) -> List[Optional[DetectorDataLoader]]:
    """Loads the data from the csv file.

    The csv format is shown in
    https://cloud.google.com/vision/automl/object-detection/docs/csv-format. We
    supports bounding box with 2 vertices for now. We support the files in the
    local machine as well.

    Args:
      filename: Name of the csv file.
      images_dir: Path to directory that store raw images. If None, the image
        path in the csv file is the path to Google Cloud Storage or the absolute
        path in the local machine.
      delimiter: Character used to separate fields.
      quotechar: Character used to quote fields containing special characters.
      num_shards: Number of shards for output file.
      max_num_images: Max number of imags to process.
      cache_dir: The cache directory to save TFRecord, metadata and json file.
        When cache_dir is None, a temporary folder will be created and will not
        be removed automatically after training which makes it can be used
        later.
      cache_prefix_filename: The cache prefix filename. If None, will
        automatically generate it based on `filename`.

    Returns:
      train_data, validation_data, test_data which are ObjectDetectorDataLoader
      objects. Can be None if without such data.
    """
    # If `cache_prefix_filename` is None, automatically generates a hash value.
    if cache_prefix_filename is None:
      cache_prefix_filename = util.get_cache_prefix_filename_from_csv(
          csv_file=filename, num_shards=num_shards)

    # Gets a list of cache files mapping `set_prefixes`.
    set_prefixes = ['TRAIN', 'VAL', 'TEST']
    cache_files_list = util.get_cache_files_sequence(
        cache_dir=cache_dir,
        cache_prefix_filename=cache_prefix_filename,
        set_prefixes=set_prefixes,
        num_shards=num_shards)

    # If not cached, writes data into tfrecord_file_paths and
    # annotations_json_file_path.
    # If `num_shards` differs, it's still not cached.
    if not util.is_all_cached(cache_files_list):
      lines_list, label_map = _group_csv_lines(
          csv_file=filename,
          set_prefixes=set_prefixes,
          delimiter=delimiter,
          quotechar=quotechar)
      cache_writer = util.CsvCacheFilesWriter(
          label_map=label_map,
          images_dir=images_dir,
          num_shards=num_shards,
          max_num_images=max_num_images)
      for cache_files, csv_lines in zip(cache_files_list, lines_list):
        if csv_lines:
          cache_writer.write_files(cache_files, csv_lines=csv_lines)

    # Loads training & validation & test data from cache.
    data = []
    for cache_files in cache_files_list:
      cache_prefix = cache_files.cache_prefix
      try:
        data.append(cls.from_cache(cache_prefix))
      except ValueError:
        # No training / validation / test data in the csv file.
        # For instance, there're only training and test data in the csv file,
        # this will make this function return `train_data, None, test_data`
        data.append(None)
    return data

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
    meta_data_file = cache_prefix + util.META_DATA_FILE_SUFFIX
    if not tf.io.gfile.exists(meta_data_file):
      raise ValueError('Metadata file %s doesn\'t exist.' % meta_data_file)
    with tf.io.gfile.GFile(meta_data_file, 'r') as f:
      meta_data = yaml.load(f, Loader=yaml.FullLoader)

    # Gets annotation json file.
    ann_json_file = cache_prefix + util.ANN_JSON_FILE_SUFFIX
    if not tf.io.gfile.exists(ann_json_file):
      ann_json_file = None

    return DataLoader(tfrecord_file_patten, meta_data['size'],
                      meta_data['label_map'], ann_json_file)

  def gen_dataset(self,
                  model_spec,
                  batch_size=None,
                  is_training=False,
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
