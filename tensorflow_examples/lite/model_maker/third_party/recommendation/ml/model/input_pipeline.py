# Lint as: python3
#   Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""Input pipeline for on-device recommendation model.

Process input TF example and prepare train/test datasets.
"""
import collections
import functools
import os
from typing import Dict, List, Tuple

import tensorflow as tf
from tensorflow_examples.lite.model_maker.third_party.recommendation.ml.configs import input_config_pb2
from tensorflow_examples.lite.model_maker.third_party.recommendation.ml.model import utils


INT_DEFAULT_VALUE = 0
STRING_DEFAULT_VALUE = 'UNK'
FLOAT_DEFAULT_VALUE = 0.0


class FeaturesAndVocabsByName(
    collections.namedtuple(
        'FeaturesAndVocabsByName', ['features_by_name', 'vocabs_by_name'])):
  """Holder for intermediate data in input processing pipeline."""
  __slots__ = ()

  def __new__(cls, features_by_name=None, vocabs_by_name=None):
    return super(FeaturesAndVocabsByName, cls).__new__(cls,
                                                       features_by_name,
                                                       vocabs_by_name)


def _prepare_feature_vocab_table(
    feature: input_config_pb2.Feature,
    vocab_file_dir: str) -> tf.lookup.StaticVocabularyTable:
  """Prepare vocabulary table for the feature if needed.

  Prepare the vocabulary table with the specified vocab_name(vocab file name)
  and vocab file directory. Feature type is required for the feature, as it
  will be used to set up the vocabulary table.

  Currently we assume the vocabulary file is a txt file and each line stores
  single feature value.

  Args:
    feature: A feature config input_config_pb2.Feature proto.
    vocab_file_dir: The directory storing vocabulary files.

  Returns:
    A vocabulary table (tf.lookup.StaticVocabularyTable) for the feature.
  """
  assert feature.HasField('vocab_name')
  assert feature.HasField('feature_type')
  assert feature.feature_type in [input_config_pb2.FeatureType.INT,
                                  input_config_pb2.FeatureType.STRING]
  vocab_path = os.path.join(vocab_file_dir, feature.vocab_name)
  num_oov_buckets = 1
  key_type = tf.string if (
      feature.feature_type == input_config_pb2.FeatureType.STRING) else tf.int64
  return tf.lookup.StaticVocabularyTable(
      tf.lookup.TextFileInitializer(
          vocab_path,
          key_dtype=key_type,
          key_index=tf.lookup.TextFileIndex.WHOLE_LINE,
          value_dtype=tf.int64,
          value_index=tf.lookup.TextFileIndex.LINE_NUMBER,
          delimiter='\t'), num_oov_buckets)


def _get_features_vocabs_for_groups(
    feature_groups: List[input_config_pb2.FeatureGroup],
    vocab_file_dir: str = '') -> FeaturesAndVocabsByName:
  """Get feature and vocabulary dictionaries for feature groups.

  Args:
    feature_groups: A list of feature group configs(
      input_config_pb2.FeatureGroup)s.
    vocab_file_dir: The directory storing vocabulary files.

  Returns:
    A FeaturesAndVocabsByName object containing features and vocabs
    dictionaries keyed feature name for features according to the feature
    group list.
  """
  features_by_name = {}
  vocabs_by_name = {}
  for feature_group in feature_groups:
    for feature in feature_group.features:
      features_by_name[feature.feature_name] = feature
      if vocab_file_dir and feature.HasField('vocab_name'):
        vocabs_by_name[feature.feature_name] = _prepare_feature_vocab_table(
            feature, vocab_file_dir)
  return FeaturesAndVocabsByName(
      features_by_name=features_by_name, vocabs_by_name=vocabs_by_name)


def get_features_and_vocabs_by_name(
    input_config: input_config_pb2.InputConfig,
    vocab_file_dir: str = '') -> FeaturesAndVocabsByName:
  """Get feature and vocabulary dictionaries according to input config.

  Args:
    input_config: The input config input_config_pb2.InputConfig proto.
    vocab_file_dir: The directory storing vocabulary files.

  Returns:
    A FeaturesAndVocabsByName object containing features and vocabs
    dictionaries keyed feature name for all features according to input
    config.
  """
  global_features_and_vocabs_by_name = (
      _get_features_vocabs_for_groups(list(input_config.global_feature_groups),
                                      vocab_file_dir))
  activity_features_and_vocabs_by_name = (
      _get_features_vocabs_for_groups(
          list(input_config.activity_feature_groups), vocab_file_dir))
  features_by_name = {
      **global_features_and_vocabs_by_name.features_by_name,
      **activity_features_and_vocabs_by_name.features_by_name
  }
  vocabs_by_name = {
      **global_features_and_vocabs_by_name.vocabs_by_name,
      **activity_features_and_vocabs_by_name.vocabs_by_name
  }
  if input_config.label_feature.HasField('vocab_name'):
    vocabs_by_name[
        input_config.label_feature.feature_name] = _prepare_feature_vocab_table(
            input_config.label_feature, vocab_file_dir)
  return FeaturesAndVocabsByName(
      features_by_name=features_by_name, vocabs_by_name=vocabs_by_name)


def _get_feature_spec(feature_type: input_config_pb2.FeatureType,
                      feature_length: int) -> tf.io.FixedLenFeature:
  """Get feature spec based on feature type and length.

  Args:
    feature_type: The type of the feature (input_config_pb2.FeatureType).
    feature_length: The length of the feature. The feature is expected to be
      a sequence.

  Returns:
    The feature spec to parse serialized examples.

  Raises:
    An error occurring if the feature type is not supported.
  """
  if feature_type == input_config_pb2.FeatureType.INT:
    dtype = tf.int64
    default_value = INT_DEFAULT_VALUE
  elif feature_type == input_config_pb2.FeatureType.STRING:
    dtype = tf.string
    default_value = STRING_DEFAULT_VALUE
  elif feature_type == input_config_pb2.FeatureType.FLOAT:
    dtype = tf.float32
    default_value = FLOAT_DEFAULT_VALUE
  else:
    raise ValueError('Unsupported feature type {}'.format(feature_type))

  return tf.io.FixedLenFeature(
      shape=[feature_length],
      dtype=dtype,
      default_value=[default_value] * feature_length)


def _get_serving_feature_spec(feature_name: str,
                              feature_type: input_config_pb2.FeatureType,
                              feature_length: int) -> tf.TensorSpec:
  if feature_type == input_config_pb2.FeatureType.INT or feature_type == input_config_pb2.FeatureType.STRING:
    dtype = tf.dtypes.int32
  elif feature_type == input_config_pb2.FeatureType.FLOAT:
    dtype = tf.dtypes.float32
  else:
    raise ValueError('Unsupported feature type {}'.format(feature_type))
  return tf.TensorSpec(shape=[feature_length], dtype=dtype, name=feature_name)


def get_serving_input_specs(
    input_config: input_config_pb2.InputConfig) -> Dict[str, tf.TensorSpec]:
  features_by_name = get_features_and_vocabs_by_name(
      input_config).features_by_name
  input_specs = collections.OrderedDict()
  for feature_name, feature in sorted(features_by_name.items()):
    input_specs[feature_name] = _get_serving_feature_spec(
        feature_name, feature.feature_type, feature.feature_length)
  return input_specs


def decode_example(
    serialized_proto: str, features_and_vocabs_by_name: FeaturesAndVocabsByName,
    label_feature_name: str) -> Tuple[Dict[str, tf.Tensor], tf.Tensor]:
  """Decode single serialized example.

  Decode single serialized example, accoring to specified features in input
  config. Perform vocabulary lookup if vocabulary is specified.

  Args:
    serialized_proto: The serialized proto that needs to be decoded.
    features_and_vocabs_by_name: A FeaturesAndVocabsByName object containing
      features and vocabs dictionaries by feature names.
    label_feature_name: Name of the label feature.

  Returns:
    features: The decoded features dictionary.
    label_feature: The label feature.

  """
  features_by_name = features_and_vocabs_by_name.features_by_name
  vocabs_by_name = features_and_vocabs_by_name.vocabs_by_name
  name_to_features = {}
  name_to_features[label_feature_name] = tf.io.FixedLenFeature([1], tf.int64)
  for feature_name, feature in features_by_name.items():
    name_to_features[feature_name] = _get_feature_spec(
        feature_type=feature.feature_type,
        feature_length=feature.feature_length)
  record_features = tf.io.parse_single_example(serialized_proto,
                                               name_to_features)

  features = {}
  for feature_name, feature_value in record_features.items():
    if feature_name in vocabs_by_name:
      features[feature_name] = vocabs_by_name[feature_name].lookup(
          feature_value)
    else:
      features[feature_name] = feature_value
  features = {
      k: v if v.dtype != tf.int64 else tf.cast(v, tf.int32)
      for k, v in features.items()
  }
  label_feature = features[label_feature_name]
  return features, label_feature


def get_input_dataset(data_filepattern: str,
                      input_config: input_config_pb2.InputConfig,
                      vocab_file_dir: str,
                      batch_size: int) -> tf.data.Dataset:
  """An input_fn to create input datasets.

  Args:
    data_filepattern: The file pattern of the input data.
    input_config: The input config input_config_pb2.InputConfig proto.
    vocab_file_dir: The path to the directory storing the vocabulary files.
    batch_size: Batch size of to-be generated dataset.

  Returns:
    A Dataset where each element is a batch of feature dicts.
  """
  features_and_vocabs_by_name = get_features_and_vocabs_by_name(
      input_config, vocab_file_dir)
  if not input_config.HasField('label_feature'):
    raise ValueError('Field label_feature is required.')
  input_files = utils.GetShardFilenames(data_filepattern)
  d = tf.data.TFRecordDataset(input_files)
  d = d.shuffle(len(input_files))
  d = d.repeat()
  d = d.shuffle(buffer_size=10000)
  d = d.map(
      functools.partial(
          decode_example,
          features_and_vocabs_by_name=features_and_vocabs_by_name,
          label_feature_name=input_config.label_feature.feature_name),
      num_parallel_calls=8)
  d = d.batch(batch_size, drop_remainder=True)
  d = d.prefetch(1)
  return d
