# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Text Model specification."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools
import logging
import os
import re
import tempfile

import tensorflow as tf
from tensorflow_examples.lite.model_maker.core import compat
from tensorflow_examples.lite.model_maker.core import file_util
from tensorflow_examples.lite.model_maker.core.api import mm_export
from tensorflow_examples.lite.model_maker.core.task import configs
from tensorflow_examples.lite.model_maker.core.task import hub_loader
from tensorflow_examples.lite.model_maker.core.task import model_util
from tensorflow_examples.lite.model_maker.core.task.model_spec import util

import tensorflow_hub as hub
from tensorflow_hub import registry
from official.nlp import optimization
from official.nlp.bert import configs as bert_configs
from official.nlp.bert import run_squad_helper
from official.nlp.bert import squad_evaluate_v1_1
from official.nlp.bert import squad_evaluate_v2_0
from official.nlp.bert import tokenization
from official.nlp.data import classifier_data_lib
from official.nlp.data import squad_lib
from official.nlp.modeling import models
# pylint: disable=g-import-not-at-top,bare-except
try:
  from official.common import distribute_utils
except:
  from official.utils.misc import distribution_utils as distribute_utils
# pylint: enable=g-import-not-at-top,bare-except


@mm_export('text_classifier.AverageWordVecSpec')
class AverageWordVecModelSpec(object):
  """A specification of averaging word vector model."""
  PAD = '<PAD>'  # Index: 0
  START = '<START>'  # Index: 1
  UNKNOWN = '<UNKNOWN>'  # Index: 2

  compat_tf_versions = compat.get_compat_tf_versions(2)
  need_gen_vocab = True
  convert_from_saved_model_tf2 = False

  def __init__(self,
               num_words=10000,
               seq_len=256,
               wordvec_dim=16,
               lowercase=True,
               dropout_rate=0.2,
               name='AverageWordVec',
               default_training_epochs=2,
               default_batch_size=32,
               model_dir=None):
    """Initialze a instance with preprocessing and model paramaters.

    Args:
      num_words: Number of words to generate the vocabulary from data.
      seq_len: Length of the sequence to feed into the model.
      wordvec_dim: Dimension of the word embedding.
      lowercase: Whether to convert all uppercase character to lowercase during
        preprocessing.
      dropout_rate: The rate for dropout.
      name: Name of the object.
      default_training_epochs: Default training epochs for training.
      default_batch_size: Default batch size for training.
      model_dir: The location of the model checkpoint files.
    """
    self.num_words = num_words
    self.seq_len = seq_len
    self.wordvec_dim = wordvec_dim
    self.lowercase = lowercase
    self.dropout_rate = dropout_rate
    self.name = name
    self.default_training_epochs = default_training_epochs
    self.default_batch_size = default_batch_size

    self.model_dir = model_dir
    if self.model_dir is None:
      self.model_dir = tempfile.mkdtemp()

  def get_name_to_features(self):
    """Gets the dictionary describing the features."""
    name_to_features = {
        'input_ids': tf.io.FixedLenFeature([self.seq_len], tf.int64),
        'label_ids': tf.io.FixedLenFeature([], tf.int64),
    }
    return name_to_features

  def select_data_from_record(self, record):
    """Dispatches records to features and labels."""
    x = record['input_ids']
    y = record['label_ids']
    return (x, y)

  def convert_examples_to_features(self, examples, tfrecord_file, label_names):
    """Converts examples to features and write them into TFRecord file."""
    writer = tf.io.TFRecordWriter(tfrecord_file)

    label_to_id = dict((name, i) for i, name in enumerate(label_names))
    for example in examples:
      features = collections.OrderedDict()

      input_ids = self.preprocess(example.text_a)
      label_id = label_to_id[example.label]
      features['input_ids'] = util.create_int_feature(input_ids)
      features['label_ids'] = util.create_int_feature([label_id])
      tf_example = tf.train.Example(
          features=tf.train.Features(feature=features))
      writer.write(tf_example.SerializeToString())
    writer.close()

  def create_model(self,
                   num_classes,
                   optimizer='rmsprop',
                   with_loss_and_metrics=True):
    """Creates the keras model."""
    # Gets a classifier model.
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=[self.seq_len], dtype=tf.int32),
        tf.keras.layers.Embedding(
            len(self.vocab), self.wordvec_dim, input_length=self.seq_len),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(self.wordvec_dim, activation=tf.nn.relu),
        tf.keras.layers.Dropout(self.dropout_rate),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    if with_loss_and_metrics:
      # Add loss and metrics in the keras model.
      model.compile(
          optimizer=optimizer,
          loss='sparse_categorical_crossentropy',
          metrics=['accuracy'])

    return model

  def run_classifier(self, train_ds, validation_ds, epochs, steps_per_epoch,
                     num_classes, **kwargs):
    """Creates classifier and runs the classifier training."""
    if epochs is None:
      epochs = self.default_training_epochs

    model = self.create_model(num_classes)

    # Trains the models.
    model.fit(
        train_ds,
        epochs=epochs,
        validation_data=validation_ds,
        steps_per_epoch=steps_per_epoch,
        **kwargs)

    return model

  def gen_vocab(self, examples):
    """Generates vocabulary list in `examples` with maximum `num_words` words."""
    vocab_counter = collections.Counter()

    for example in examples:
      tokens = self._tokenize(example.text_a)
      for token in tokens:
        vocab_counter[token] += 1

    self.vocab_freq = vocab_counter.most_common(self.num_words)
    vocab_list = [self.PAD, self.START, self.UNKNOWN
                 ] + [word for word, _ in self.vocab_freq]
    self.vocab = collections.OrderedDict(
        ((v, i) for i, v in enumerate(vocab_list)))
    return self.vocab

  def preprocess(self, raw_text):
    """Preprocess the text for text classification."""
    tokens = self._tokenize(raw_text)

    # Gets ids for START, PAD and UNKNOWN tokens.
    start_id = self.vocab[self.START]
    pad_id = self.vocab[self.PAD]
    unknown_id = self.vocab[self.UNKNOWN]

    token_ids = [self.vocab.get(token, unknown_id) for token in tokens]
    token_ids = [start_id] + token_ids

    if len(token_ids) < self.seq_len:
      # Padding.
      pad_length = self.seq_len - len(token_ids)
      token_ids = token_ids + pad_length * [pad_id]
    else:
      token_ids = token_ids[:self.seq_len]

    return token_ids

  def _tokenize(self, text):
    r"""Splits by '\W' except '\''."""
    text = tf.compat.as_text(text)
    if self.lowercase:
      text = text.lower()
    tokens = re.compile(r'[^\w\']+').split(text.strip())
    return list(filter(None, tokens))

  def save_vocab(self, vocab_filename):
    """Saves the vocabulary in `vocab_filename`."""
    with tf.io.gfile.GFile(vocab_filename, 'w') as f:
      for token, index in self.vocab.items():
        f.write('%s %d\n' % (token, index))

    tf.compat.v1.logging.info('Saved vocabulary in %s.', vocab_filename)

  def load_vocab(self, vocab_filename):
    """Loads vocabulary from `vocab_filename`."""
    with tf.io.gfile.GFile(vocab_filename, 'r') as f:
      vocab_list = []
      for line in f:
        word, index = line.strip().split()
        vocab_list.append((word, int(index)))
    self.vocab = collections.OrderedDict(vocab_list)
    return self.vocab

  def get_config(self):
    """Gets the configuration."""
    return {
        'num_words': self.num_words,
        'seq_len': self.seq_len,
        'wordvec_dim': self.wordvec_dim,
        'lowercase': self.lowercase
    }

  def get_default_quantization_config(self):
    """Gets the default quantization configuration."""
    return None


def create_classifier_model(bert_config,
                            num_labels,
                            max_seq_length,
                            initializer=None,
                            hub_module_url=None,
                            hub_module_trainable=True,
                            is_tf2=True):
  """BERT classifier model in functional API style.

  Construct a Keras model for predicting `num_labels` outputs from an input with
  maximum sequence length `max_seq_length`.

  Args:
    bert_config: BertConfig, the config defines the core Bert model.
    num_labels: integer, the number of classes.
    max_seq_length: integer, the maximum input sequence length.
    initializer: Initializer for the final dense layer in the span labeler.
      Defaulted to TruncatedNormal initializer.
    hub_module_url: TF-Hub path/url to Bert module.
    hub_module_trainable: True to finetune layers in the hub module.
    is_tf2: boolean, whether the hub module is in TensorFlow 2.x format.

  Returns:
    Combined prediction model (words, mask, type) -> (one-hot labels)
    BERT sub-model (words, mask, type) -> (bert_outputs)
  """
  if initializer is None:
    initializer = tf.keras.initializers.TruncatedNormal(
        stddev=bert_config.initializer_range)

  input_word_ids = tf.keras.layers.Input(
      shape=(max_seq_length,), dtype=tf.int32, name='input_word_ids')
  input_mask = tf.keras.layers.Input(
      shape=(max_seq_length,), dtype=tf.int32, name='input_mask')
  input_type_ids = tf.keras.layers.Input(
      shape=(max_seq_length,), dtype=tf.int32, name='input_type_ids')

  if is_tf2:
    bert_model = hub.KerasLayer(hub_module_url, trainable=hub_module_trainable)
    pooled_output, _ = bert_model([input_word_ids, input_mask, input_type_ids])
  else:
    bert_model = hub_loader.HubKerasLayerV1V2(
        hub_module_url,
        signature='tokens',
        output_key='pooled_output',
        trainable=hub_module_trainable)

    pooled_output = bert_model({
        'input_ids': input_word_ids,
        'input_mask': input_mask,
        'segment_ids': input_type_ids
    })

  output = tf.keras.layers.Dropout(rate=bert_config.hidden_dropout_prob)(
      pooled_output)
  output = tf.keras.layers.Dense(
      num_labels,
      kernel_initializer=initializer,
      name='output',
      activation='softmax',
      dtype=tf.float32)(
          output)

  return tf.keras.Model(
      inputs=[input_word_ids, input_mask, input_type_ids],
      outputs=output), bert_model


class BertModelSpec(object):
  """A specification of BERT model."""

  compat_tf_versions = compat.get_compat_tf_versions(2)
  need_gen_vocab = False
  convert_from_saved_model_tf2 = True  # Convert to TFLite from saved_model.

  def __init__(
      self,
      uri='https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1',
      model_dir=None,
      seq_len=128,
      dropout_rate=0.1,
      initializer_range=0.02,
      learning_rate=3e-5,
      distribution_strategy='mirrored',
      num_gpus=-1,
      tpu='',
      trainable=True,
      do_lower_case=True,
      is_tf2=True,
      name='Bert',
      tflite_input_name=None,
      default_batch_size=32):
    """Initialze an instance with model paramaters.

    Args:
      uri: TF-Hub path/url to Bert module.
      model_dir: The location of the model checkpoint files.
      seq_len: Length of the sequence to feed into the model.
      dropout_rate: The rate for dropout.
      initializer_range: The stdev of the truncated_normal_initializer for
        initializing all weight matrices.
      learning_rate: The initial learning rate for Adam.
      distribution_strategy:  A string specifying which distribution strategy to
        use. Accepted values are 'off', 'one_device', 'mirrored',
        'parameter_server', 'multi_worker_mirrored', and 'tpu' -- case
        insensitive. 'off' means not to use Distribution Strategy; 'tpu' means
        to use TPUStrategy using `tpu_address`.
      num_gpus: How many GPUs to use at each worker with the
        DistributionStrategies API. The default is -1, which means utilize all
        available GPUs.
      tpu: TPU address to connect to.
      trainable: boolean, whether pretrain layer is trainable.
      do_lower_case: boolean, whether to lower case the input text. Should be
        True for uncased models and False for cased models.
      is_tf2: boolean, whether the hub module is in TensorFlow 2.x format.
      name: The name of the object.
      tflite_input_name: Dict, input names for the TFLite model.
      default_batch_size: Default batch size for training.
    """
    if compat.get_tf_behavior() not in self.compat_tf_versions:
      raise ValueError('Incompatible versions. Expect {}, but got {}.'.format(
          self.compat_tf_versions, compat.get_tf_behavior()))
    self.seq_len = seq_len
    self.dropout_rate = dropout_rate
    self.initializer_range = initializer_range
    self.learning_rate = learning_rate
    self.trainable = trainable

    self.model_dir = model_dir
    if self.model_dir is None:
      self.model_dir = tempfile.mkdtemp()

    num_gpus = util.get_num_gpus(num_gpus)
    self.strategy = distribute_utils.get_distribution_strategy(
        distribution_strategy=distribution_strategy,
        num_gpus=num_gpus,
        tpu_address=tpu)
    self.tpu = tpu
    self.uri = uri
    self.do_lower_case = do_lower_case
    self.is_tf2 = is_tf2

    self.bert_config = bert_configs.BertConfig(
        0,
        initializer_range=self.initializer_range,
        hidden_dropout_prob=self.dropout_rate)

    self.is_built = False
    self.name = name

    if tflite_input_name is None:
      tflite_input_name = {
          'ids': 'serving_default_input_word_ids:0',
          'mask': 'serving_default_input_mask:0',
          'segment_ids': 'serving_default_input_type_ids:0'
      }
    self.tflite_input_name = tflite_input_name
    self.default_batch_size = default_batch_size

  def get_default_quantization_config(self):
    """Gets the default quantization configuration."""
    config = configs.QuantizationConfig.for_dynamic()
    config.experimental_new_quantizer = True
    return config

  def reorder_input_details(self, tflite_input_details):
    """Reorders the tflite input details to map the order of keras model."""
    for detail in tflite_input_details:
      name = detail['name']
      if 'input_word_ids' in name:
        input_word_ids_detail = detail
      elif 'input_mask' in name:
        input_mask_detail = detail
      elif 'input_type_ids' in name:
        input_type_ids_detail = detail
    return [input_word_ids_detail, input_mask_detail, input_type_ids_detail]

  def build(self):
    """Builds the class. Used for lazy initialization."""
    if self.is_built:
      return
    self.vocab_file = os.path.join(
        registry.resolver(self.uri), 'assets', 'vocab.txt')
    self.tokenizer = tokenization.FullTokenizer(self.vocab_file,
                                                self.do_lower_case)

  def save_vocab(self, vocab_filename):
    """Prints the file path to the vocabulary."""
    if not self.is_built:
      self.build()
    tf.io.gfile.copy(self.vocab_file, vocab_filename, overwrite=True)
    tf.compat.v1.logging.info('Saved vocabulary in %s.', vocab_filename)


@mm_export('text_classifier.BertClassifierSpec')
class BertClassifierModelSpec(BertModelSpec):
  """A specification of BERT model for text classification."""

  def get_name_to_features(self):
    """Gets the dictionary describing the features."""
    name_to_features = {
        'input_ids': tf.io.FixedLenFeature([self.seq_len], tf.int64),
        'input_mask': tf.io.FixedLenFeature([self.seq_len], tf.int64),
        'segment_ids': tf.io.FixedLenFeature([self.seq_len], tf.int64),
        'label_ids': tf.io.FixedLenFeature([], tf.int64),
        'is_real_example': tf.io.FixedLenFeature([], tf.int64),
    }
    return name_to_features

  def select_data_from_record(self, record):
    """Dispatches records to features and labels."""
    x = {
        'input_word_ids': record['input_ids'],
        'input_mask': record['input_mask'],
        'input_type_ids': record['segment_ids']
    }
    y = record['label_ids']
    return (x, y)

  def convert_examples_to_features(self, examples, tfrecord_file, label_names):
    """Converts examples to features and write them into TFRecord file."""
    if not self.is_built:
      self.build()
    classifier_data_lib.file_based_convert_examples_to_features(
        examples, label_names, self.seq_len, self.tokenizer, tfrecord_file)

  def create_model(self,
                   num_classes,
                   optimizer='adam',
                   with_loss_and_metrics=True):
    """Creates the keras model."""
    bert_model, _ = create_classifier_model(
        self.bert_config,
        num_classes,
        self.seq_len,
        hub_module_url=self.uri,
        hub_module_trainable=self.trainable,
        is_tf2=self.is_tf2)

    # Defines evaluation metrics function, which will create metrics in the
    # correct device and strategy scope.
    def metric_fn():
      return tf.keras.metrics.SparseCategoricalAccuracy(
          'test_accuracy', dtype=tf.float32)

    if with_loss_and_metrics:
      # Add loss and metrics in the keras model.
      bert_model.compile(
          optimizer=optimizer,
          loss=tf.keras.losses.SparseCategoricalCrossentropy(),
          metrics=[metric_fn()])

    return bert_model

  def run_classifier(self, train_ds, validation_ds, epochs, steps_per_epoch,
                     num_classes, **kwargs):
    """Creates classifier and runs the classifier training.

    Args:
      train_ds: tf.data.Dataset, training data to be fed in
        tf.keras.Model.fit().
      validation_ds: tf.data.Dataset, validation data to be fed in
        tf.keras.Model.fit().
      epochs: Integer, training epochs.
      steps_per_epoch: Integer or None. Total number of steps (batches of
        samples) before declaring one epoch finished and starting the next
        epoch. If `steps_per_epoch` is None, the epoch will run until the input
        dataset is exhausted.
      num_classes: Interger, number of classes.
      **kwargs: Other parameters used in the tf.keras.Model.fit().

    Returns:
      tf.keras.Model, the keras model that's already trained.
    """
    if steps_per_epoch is None:
      logging.info(
          'steps_per_epoch is None, use %d as the estimated steps_per_epoch',
          model_util.ESTIMITED_STEPS_PER_EPOCH)
      steps_per_epoch = model_util.ESTIMITED_STEPS_PER_EPOCH
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(total_steps * 0.1)
    initial_lr = self.learning_rate

    with distribute_utils.get_strategy_scope(self.strategy):
      optimizer = optimization.create_optimizer(initial_lr, total_steps,
                                                warmup_steps)
      bert_model = self.create_model(num_classes, optimizer)

    bert_model.fit(
        x=train_ds, validation_data=validation_ds, epochs=epochs, **kwargs)

    return bert_model

  def get_config(self):
    """Gets the configuration."""
    # Only preprocessing related variables are included.
    return {'uri': self.uri, 'seq_len': self.seq_len}


def dump_to_files(all_predictions, all_nbest_json, scores_diff_json,
                  version_2_with_negative, output_dir):
  """Save output to json files for question answering."""
  output_prediction_file = os.path.join(output_dir, 'predictions.json')
  output_nbest_file = os.path.join(output_dir, 'nbest_predictions.json')
  output_null_log_odds_file = os.path.join(output_dir, 'null_odds.json')
  tf.compat.v1.logging.info('Writing predictions to: %s',
                            (output_prediction_file))
  tf.compat.v1.logging.info('Writing nbest to: %s', (output_nbest_file))

  squad_lib.write_to_json_files(all_predictions, output_prediction_file)
  squad_lib.write_to_json_files(all_nbest_json, output_nbest_file)
  if version_2_with_negative:
    squad_lib.write_to_json_files(scores_diff_json, output_null_log_odds_file)


def create_qa_model(bert_config,
                    max_seq_length,
                    initializer=None,
                    hub_module_url=None,
                    hub_module_trainable=True,
                    is_tf2=True):
  """Returns BERT qa model along with core BERT model to import weights.

  Args:
    bert_config: BertConfig, the config defines the core Bert model.
    max_seq_length: integer, the maximum input sequence length.
    initializer: Initializer for the final dense layer in the span labeler.
      Defaulted to TruncatedNormal initializer.
    hub_module_url: TF-Hub path/url to Bert module.
    hub_module_trainable: True to finetune layers in the hub module.
    is_tf2: boolean, whether the hub module is in TensorFlow 2.x format.

  Returns:
    A tuple of (1) keras model that outputs start logits and end logits and
    (2) the core BERT transformer encoder.
  """

  if initializer is None:
    initializer = tf.keras.initializers.TruncatedNormal(
        stddev=bert_config.initializer_range)

  input_word_ids = tf.keras.layers.Input(
      shape=(max_seq_length,), dtype=tf.int32, name='input_word_ids')
  input_mask = tf.keras.layers.Input(
      shape=(max_seq_length,), dtype=tf.int32, name='input_mask')
  input_type_ids = tf.keras.layers.Input(
      shape=(max_seq_length,), dtype=tf.int32, name='input_type_ids')

  if is_tf2:
    core_model = hub.KerasLayer(hub_module_url, trainable=hub_module_trainable)
    pooled_output, sequence_output = core_model(
        [input_word_ids, input_mask, input_type_ids])
  else:
    bert_model = hub_loader.HubKerasLayerV1V2(
        hub_module_url,
        signature='tokens',
        signature_outputs_as_dict=True,
        trainable=hub_module_trainable)
    outputs = bert_model({
        'input_ids': input_word_ids,
        'input_mask': input_mask,
        'segment_ids': input_type_ids
    })

    pooled_output = outputs['pooled_output']
    sequence_output = outputs['sequence_output']

  bert_encoder = tf.keras.Model(
      inputs=[input_word_ids, input_mask, input_type_ids],
      outputs=[sequence_output, pooled_output],
      name='core_model')
  return models.BertSpanLabeler(
      network=bert_encoder, initializer=initializer), bert_encoder


def create_qa_model_from_squad(max_seq_length,
                               hub_module_url,
                               hub_module_trainable=True,
                               is_tf2=False):
  """Creates QA model the initialized from the model retrained on Squad dataset.

  Args:
    max_seq_length: integer, the maximum input sequence length.
    hub_module_url: TF-Hub path/url to Bert module that's retrained on Squad
      dataset.
    hub_module_trainable: True to finetune layers in the hub module.
    is_tf2: boolean, whether the hub module is in TensorFlow 2.x format.

  Returns:
    Keras model that outputs start logits and end logits.
  """
  if is_tf2:
    raise ValueError('Only supports to load TensorFlow 1.x hub module.')

  input_word_ids = tf.keras.layers.Input(
      shape=(max_seq_length,), dtype=tf.int32, name='input_word_ids')
  input_mask = tf.keras.layers.Input(
      shape=(max_seq_length,), dtype=tf.int32, name='input_mask')
  input_type_ids = tf.keras.layers.Input(
      shape=(max_seq_length,), dtype=tf.int32, name='input_type_ids')

  squad_bert = hub_loader.HubKerasLayerV1V2(
      hub_module_url,
      signature='squad',
      signature_outputs_as_dict=True,
      trainable=hub_module_trainable)

  outputs = squad_bert({
      'input_ids': input_word_ids,
      'input_mask': input_mask,
      'segment_ids': input_type_ids
  })
  start_logits = tf.keras.layers.Lambda(
      tf.identity, name='start_positions')(
          outputs['start_logits'])
  end_logits = tf.keras.layers.Lambda(
      tf.identity, name='end_positions')(
          outputs['end_logits'])

  return tf.keras.Model(
      inputs=[input_word_ids, input_mask, input_type_ids],
      outputs=[start_logits, end_logits])


@mm_export('question_answer.BertQaSpec')
class BertQAModelSpec(BertModelSpec):
  """A specification of BERT model for question answering."""

  def __init__(
      self,
      uri='https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1',
      model_dir=None,
      seq_len=384,
      query_len=64,
      doc_stride=128,
      dropout_rate=0.1,
      initializer_range=0.02,
      learning_rate=8e-5,
      distribution_strategy='mirrored',
      num_gpus=-1,
      tpu='',
      trainable=True,
      predict_batch_size=8,
      do_lower_case=True,
      is_tf2=True,
      tflite_input_name=None,
      tflite_output_name=None,
      init_from_squad_model=False,
      default_batch_size=16,
      name='Bert'):
    """Initialze an instance with model paramaters.

    Args:
      uri: TF-Hub path/url to Bert module.
      model_dir: The location of the model checkpoint files.
      seq_len: Length of the sequence to feed into the model.
      query_len: Length of the query to feed into the model.
      doc_stride: The stride when we do a sliding window approach to take chunks
        of the documents.
      dropout_rate: The rate for dropout.
      initializer_range: The stdev of the truncated_normal_initializer for
        initializing all weight matrices.
      learning_rate: The initial learning rate for Adam.
      distribution_strategy:  A string specifying which distribution strategy to
        use. Accepted values are 'off', 'one_device', 'mirrored',
        'parameter_server', 'multi_worker_mirrored', and 'tpu' -- case
        insensitive. 'off' means not to use Distribution Strategy; 'tpu' means
        to use TPUStrategy using `tpu_address`.
      num_gpus: How many GPUs to use at each worker with the
        DistributionStrategies API. The default is -1, which means utilize all
        available GPUs.
      tpu: TPU address to connect to.
      trainable: boolean, whether pretrain layer is trainable.
      predict_batch_size: Batch size for prediction.
      do_lower_case: boolean, whether to lower case the input text. Should be
        True for uncased models and False for cased models.
      is_tf2: boolean, whether the hub module is in TensorFlow 2.x format.
      tflite_input_name: Dict, input names for the TFLite model.
      tflite_output_name: Dict, output names for the TFLite model.
      init_from_squad_model: boolean, whether to initialize from the model that
        is already retrained on Squad 1.1.
      default_batch_size: Default batch size for training.
      name: Name of the object.
    """
    super(BertQAModelSpec,
          self).__init__(uri, model_dir, seq_len, dropout_rate,
                         initializer_range, learning_rate,
                         distribution_strategy, num_gpus, tpu, trainable,
                         do_lower_case, is_tf2, name, tflite_input_name,
                         default_batch_size)
    self.query_len = query_len
    self.doc_stride = doc_stride
    self.predict_batch_size = predict_batch_size
    if tflite_output_name is None:
      tflite_output_name = {
          'start_logits': 'StatefulPartitionedCall:1',
          'end_logits': 'StatefulPartitionedCall:0'
      }
    self.tflite_output_name = tflite_output_name
    self.init_from_squad_model = init_from_squad_model

  def get_name_to_features(self, is_training):
    """Gets the dictionary describing the features."""
    name_to_features = {
        'input_ids': tf.io.FixedLenFeature([self.seq_len], tf.int64),
        'input_mask': tf.io.FixedLenFeature([self.seq_len], tf.int64),
        'segment_ids': tf.io.FixedLenFeature([self.seq_len], tf.int64),
    }

    if is_training:
      name_to_features['start_positions'] = tf.io.FixedLenFeature([], tf.int64)
      name_to_features['end_positions'] = tf.io.FixedLenFeature([], tf.int64)
    else:
      name_to_features['unique_ids'] = tf.io.FixedLenFeature([], tf.int64)

    return name_to_features

  def select_data_from_record(self, record):
    """Dispatches records to features and labels."""
    x, y = {}, {}
    for name, tensor in record.items():
      if name in ('start_positions', 'end_positions'):
        y[name] = tensor
      elif name == 'input_ids':
        x['input_word_ids'] = tensor
      elif name == 'segment_ids':
        x['input_type_ids'] = tensor
      else:
        x[name] = tensor
    return (x, y)

  def get_config(self):
    """Gets the configuration."""
    # Only preprocessing related variables are included.
    return {
        'uri': self.uri,
        'seq_len': self.seq_len,
        'query_len': self.query_len,
        'doc_stride': self.doc_stride
    }

  def convert_examples_to_features(self, examples, is_training, output_fn,
                                   batch_size):
    """Converts examples to features and write them into TFRecord file."""
    if not self.is_built:
      self.build()

    return squad_lib.convert_examples_to_features(
        examples=examples,
        tokenizer=self.tokenizer,
        max_seq_length=self.seq_len,
        doc_stride=self.doc_stride,
        max_query_length=self.query_len,
        is_training=is_training,
        output_fn=output_fn,
        batch_size=batch_size)

  def create_model(self):
    """Creates the model for qa task."""
    if self.init_from_squad_model:
      return create_qa_model_from_squad(self.seq_len, self.uri, self.trainable,
                                        self.is_tf2)
    else:
      qa_model, _ = create_qa_model(
          self.bert_config,
          self.seq_len,
          hub_module_url=self.uri,
          hub_module_trainable=self.trainable,
          is_tf2=self.is_tf2)
      return qa_model

  def train(self, train_ds, epochs, steps_per_epoch, **kwargs):
    """Run bert QA training.

    Args:
      train_ds: tf.data.Dataset, training data to be fed in
        tf.keras.Model.fit().
      epochs: Integer, training epochs.
      steps_per_epoch: Integer or None. Total number of steps (batches of
        samples) before declaring one epoch finished and starting the next
        epoch. If `steps_per_epoch` is None, the epoch will run until the input
        dataset is exhausted.
      **kwargs: Other parameters used in the tf.keras.Model.fit().

    Returns:
      tf.keras.Model, the keras model that's already trained.
    """
    if steps_per_epoch is None:
      logging.info(
          'steps_per_epoch is None, use %d as the estimated steps_per_epoch',
          model_util.ESTIMITED_STEPS_PER_EPOCH)
      steps_per_epoch = model_util.ESTIMITED_STEPS_PER_EPOCH
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(total_steps * 0.1)

    def _loss_fn(positions, logits):
      """Get losss function for QA model."""
      loss = tf.keras.losses.sparse_categorical_crossentropy(
          positions, logits, from_logits=True)
      return tf.reduce_mean(loss)

    with distribute_utils.get_strategy_scope(self.strategy):
      bert_model = self.create_model()
      optimizer = optimization.create_optimizer(self.learning_rate, total_steps,
                                                warmup_steps)

      bert_model.compile(
          optimizer=optimizer, loss=_loss_fn, loss_weights=[0.5, 0.5])

    if not bert_model.trainable_variables:
      tf.compat.v1.logging.warning(
          'Trainable variables in the model are empty.')
      return bert_model

    bert_model.fit(x=train_ds, epochs=epochs, **kwargs)

    return bert_model

  def _predict(self, model, dataset, num_steps):
    """Predicts the dataset using distribute strategy."""
    # TODO(wangtz): We should probably set default strategy as self.strategy
    # if not specified.
    strategy = self.strategy or tf.distribute.get_strategy()
    predict_iterator = iter(strategy.experimental_distribute_dataset(dataset))

    @tf.function
    def predict_step(iterator):
      """Predicts on distributed devices."""

      def _replicated_step(inputs):
        """Replicated prediction calculation."""
        x, _ = inputs
        unique_ids = x.pop('unique_ids')
        start_logits, end_logits = model(x, training=False)
        return dict(
            unique_ids=unique_ids,
            start_logits=start_logits,
            end_logits=end_logits)

      outputs = strategy.run(_replicated_step, args=(next(iterator),))
      return tf.nest.map_structure(strategy.experimental_local_results, outputs)

    all_results = []
    for _ in range(num_steps):
      predictions = predict_step(predict_iterator)
      for result in run_squad_helper.get_raw_results(predictions):
        all_results.append(result)
      if len(all_results) % 100 == 0:
        tf.compat.v1.logging.info('Made predictions for %d records.',
                                  len(all_results))
    return all_results

  def predict(self, model, dataset, num_steps):
    """Predicts the dataset for `model`."""
    return self._predict(model, dataset, num_steps)

  def reorder_output_details(self, tflite_output_details):
    """Reorders the tflite output details to map the order of keras model."""
    for detail in tflite_output_details:
      name = detail['name']
      if self.tflite_output_name['start_logits'] == name:
        start_logits_detail = detail
      if self.tflite_output_name['end_logits'] == name:
        end_logits_detail = detail
    return (start_logits_detail, end_logits_detail)

  def predict_tflite(self, tflite_filepath, dataset):
    """Predicts the dataset for TFLite model in `tflite_filepath`."""
    all_results = []
    lite_runner = model_util.LiteRunner(tflite_filepath,
                                        self.reorder_input_details,
                                        self.reorder_output_details)
    for features, _ in dataset:
      outputs = lite_runner.run(features)
      for unique_id, start_logits, end_logits in zip(features['unique_ids'],
                                                     outputs[0], outputs[1]):
        raw_result = run_squad_helper.RawResult(
            unique_id=unique_id.numpy(),
            start_logits=start_logits.tolist(),
            end_logits=end_logits.tolist())
        all_results.append(raw_result)
        if len(all_results) % 100 == 0:
          tf.compat.v1.logging.info('Made predictions for %d records.',
                                    len(all_results))
    return all_results

  def evaluate(self, model, tflite_filepath, dataset, num_steps, eval_examples,
               eval_features, predict_file, version_2_with_negative,
               max_answer_length, null_score_diff_threshold, verbose_logging,
               output_dir):
    """Evaluate QA model.

    Args:
      model: The keras model to be evaluated.
      tflite_filepath: File path to the TFLite model.
      dataset: tf.data.Dataset used for evaluation.
      num_steps: Number of steps to evaluate the model.
      eval_examples: List of `squad_lib.SquadExample` for evaluation data.
      eval_features: List of `squad_lib.InputFeatures` for evaluation data.
      predict_file: The input predict file.
      version_2_with_negative: Whether the input predict file is SQuAD 2.0
        format.
      max_answer_length: The maximum length of an answer that can be generated.
        This is needed because the start and end predictions are not conditioned
        on one another.
      null_score_diff_threshold: If null_score - best_non_null is greater than
        the threshold, predict null. This is only used for SQuAD v2.
      verbose_logging: If true, all of the warnings related to data processing
        will be printed. A number of warnings are expected for a normal SQuAD
        evaluation.
      output_dir: The output directory to save output to json files:
        predictions.json, nbest_predictions.json, null_odds.json. If None, skip
        saving to json files.

    Returns:
      A dict contains two metrics: Exact match rate and F1 score.
    """
    if model is not None and tflite_filepath is not None:
      raise ValueError('Exactly one of the paramaters `model` and '
                       '`tflite_filepath` should be set.')
    elif model is None and tflite_filepath is None:
      raise ValueError('At least one of the parameters `model` and '
                       '`tflite_filepath` are None.')

    if tflite_filepath is not None:
      all_results = self.predict_tflite(tflite_filepath, dataset)
    else:
      all_results = self.predict(model, dataset, num_steps)

    all_predictions, all_nbest_json, scores_diff_json = (
        squad_lib.postprocess_output(
            eval_examples,
            eval_features,
            all_results,
            n_best_size=20,
            max_answer_length=max_answer_length,
            do_lower_case=self.do_lower_case,
            version_2_with_negative=version_2_with_negative,
            null_score_diff_threshold=null_score_diff_threshold,
            verbose=verbose_logging))

    if output_dir is not None:
      dump_to_files(all_predictions, all_nbest_json, scores_diff_json,
                    version_2_with_negative, output_dir)

    dataset_json = file_util.load_json_file(predict_file)
    pred_dataset = dataset_json['data']

    if version_2_with_negative:
      eval_metrics = squad_evaluate_v2_0.evaluate(pred_dataset, all_predictions,
                                                  scores_diff_json)
    else:
      eval_metrics = squad_evaluate_v1_1.evaluate(pred_dataset, all_predictions)
    return eval_metrics


mobilebert_classifier_spec = functools.partial(
    BertClassifierModelSpec,
    uri='https://tfhub.dev/google/mobilebert/uncased_L-24_H-128_B-512_A-4_F-4_OPT/1',
    is_tf2=False,
    distribution_strategy='off',
    name='MobileBert',
    default_batch_size=48,
)
mobilebert_classifier_spec.__doc__ = util.wrap_doc(
    BertClassifierModelSpec,
    'Creates MobileBert model spec for the text classification task. See also: `tflite_model_maker.text_classifier.BertClassifierSpec`.'
)
mm_export('text_classifier.MobileBertClassifierSpec').export_constant(
    __name__, 'mobilebert_classifier_spec')

mobilebert_qa_spec = functools.partial(
    BertQAModelSpec,
    uri='https://tfhub.dev/google/mobilebert/uncased_L-24_H-128_B-512_A-4_F-4_OPT/1',
    is_tf2=False,
    distribution_strategy='off',
    learning_rate=4e-05,
    name='MobileBert',
    default_batch_size=32,
)
mobilebert_qa_spec.__doc__ = util.wrap_doc(
    BertQAModelSpec,
    'Creates MobileBert model spec for the question answer task. See also: `tflite_model_maker.question_answer.BertQaSpec`.'
)
mm_export('question_answer.MobileBertQaSpec').export_constant(
    __name__, 'mobilebert_qa_spec')

mobilebert_qa_squad_spec = functools.partial(
    BertQAModelSpec,
    uri='https://tfhub.dev/google/mobilebert/uncased_L-24_H-128_B-512_A-4_F-4_OPT/squadv1/1',
    is_tf2=False,
    distribution_strategy='off',
    learning_rate=4e-05,
    name='MobileBert',
    init_from_squad_model=True,
    default_batch_size=32,
)
mobilebert_qa_squad_spec.__doc__ = util.wrap_doc(
    BertQAModelSpec,
    'Creates MobileBert model spec that\'s already retrained on SQuAD1.1 for '
    'the question answer task. See also: `tflite_model_maker.question_answer.BertQaSpec`.'
)
mm_export('question_answer.MobileBertQaSquadSpec').export_constant(
    __name__, 'mobilebert_qa_squad_spec')
