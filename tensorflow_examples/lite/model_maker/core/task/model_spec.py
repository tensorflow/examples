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
"""Model specification."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import inspect
import os
import re
import tempfile

import tensorflow as tf
from tensorflow_examples.lite.model_maker.core import compat
from tensorflow_examples.lite.model_maker.core import file_util
from tensorflow_examples.lite.model_maker.core.task import hub_loader
from tensorflow_examples.lite.model_maker.core.task import model_util

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
from official.utils.misc import distribution_utils


def create_int_feature(values):
  feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
  return feature


def _get_compat_tf_versions(compat_tf_versions=None):
  """Gets compatible tf versions (default: [2]).

  Args:
    compat_tf_versions: int, int list or None, indicates compatible versions.

  Returns:
    A list of compatible tf versions.
  """
  if compat_tf_versions is None:
    compat_tf_versions = [2]
  if not isinstance(compat_tf_versions, list):
    compat_tf_versions = [compat_tf_versions]
  return compat_tf_versions


def get_num_gpus(num_gpus):
  try:
    tot_num_gpus = len(tf.config.experimental.list_physical_devices('GPU'))
  except (tf.errors.NotFoundError, tf.errors.InternalError):
    tot_num_gpus = max(0, num_gpus)
  if num_gpus > tot_num_gpus or num_gpus == -1:
    num_gpus = tot_num_gpus
  return num_gpus


class ImageModelSpec(object):
  """A specification of image model."""

  mean_rgb = [0.0]
  stddev_rgb = [255.0]

  def __init__(self,
               uri,
               compat_tf_versions=None,
               input_image_shape=None,
               name=''):
    self.uri = uri
    self.compat_tf_versions = _get_compat_tf_versions(compat_tf_versions)
    self.name = name

    if input_image_shape is None:
      input_image_shape = [224, 224]
    self.input_image_shape = input_image_shape


mobilenet_v2_spec = ImageModelSpec(
    uri='https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4',
    compat_tf_versions=2,
    name='mobilenet_v2')

resnet_50_spec = ImageModelSpec(
    uri='https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/4',
    compat_tf_versions=2,
    name='resnet_50')

efficientnet_lite0_spec = ImageModelSpec(
    uri='https://tfhub.dev/tensorflow/efficientnet/lite0/feature-vector/2',
    compat_tf_versions=[1, 2],
    name='efficientnet_lite0')

efficientnet_lite1_spec = ImageModelSpec(
    uri='https://tfhub.dev/tensorflow/efficientnet/lite1/feature-vector/2',
    compat_tf_versions=[1, 2],
    input_image_shape=[240, 240],
    name='efficientnet_lite1')

efficientnet_lite2_spec = ImageModelSpec(
    uri='https://tfhub.dev/tensorflow/efficientnet/lite2/feature-vector/2',
    compat_tf_versions=[1, 2],
    input_image_shape=[260, 260],
    name='efficientnet_lite2')

efficientnet_lite3_spec = ImageModelSpec(
    uri='https://tfhub.dev/tensorflow/efficientnet/lite3/feature-vector/2',
    compat_tf_versions=[1, 2],
    input_image_shape=[280, 280],
    name='efficientnet_lite3')

efficientnet_lite4_spec = ImageModelSpec(
    uri='https://tfhub.dev/tensorflow/efficientnet/lite4/feature-vector/2',
    compat_tf_versions=[1, 2],
    input_image_shape=[300, 300],
    name='efficientnet_lite4')


class AverageWordVecModelSpec(object):
  """A specification of averaging word vector model."""
  PAD = '<PAD>'  # Index: 0
  START = '<START>'  # Index: 1
  UNKNOWN = '<UNKNOWN>'  # Index: 2

  compat_tf_versions = _get_compat_tf_versions(2)
  need_gen_vocab = True
  default_training_epochs = 2
  default_batch_size = 32
  convert_from_saved_model_tf2 = False

  def __init__(self,
               num_words=10000,
               seq_len=256,
               wordvec_dim=16,
               lowercase=True,
               dropout_rate=0.2,
               name='AverageWordVec'):
    """Initialze a instance with preprocessing and model paramaters.

    Args:
      num_words: Number of words to generate the vocabulary from data.
      seq_len: Length of the sequence to feed into the model.
      wordvec_dim: Dimension of the word embedding.
      lowercase: Whether to convert all uppercase character to lowercase during
        preprocessing.
      dropout_rate: The rate for dropout.
      name: Name of the object.
    """
    self.num_words = num_words
    self.seq_len = seq_len
    self.wordvec_dim = wordvec_dim
    self.lowercase = lowercase
    self.dropout_rate = dropout_rate
    self.name = name

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
      features['input_ids'] = create_int_feature(input_ids)
      features['label_ids'] = create_int_feature([label_id])
      tf_example = tf.train.Example(
          features=tf.train.Features(feature=features))
      writer.write(tf_example.SerializeToString())
    writer.close()

  def create_model(self, num_classes, optimizer='rmsprop'):
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

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

    return model

  def run_classifier(self, train_input_fn, validation_input_fn, epochs,
                     steps_per_epoch, validation_steps, num_classes):
    """Creates classifier and runs the classifier training."""
    if epochs is None:
      epochs = self.default_training_epochs

    model = self.create_model(num_classes)
    # Gets training and validation dataset
    train_ds = train_input_fn()
    validation_ds = None
    if validation_input_fn is not None:
      validation_ds = validation_input_fn()

    # Trains the models.
    model.fit(
        train_ds,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=validation_ds,
        validation_steps=validation_steps)

    return model

  def gen_vocab(self, examples):
    """Generates vocabulary list in `examples` with maximum `num_words` words."""
    vocab_counter = collections.Counter()

    for example in examples:
      tokens = self._tokenize(example.text_a)
      for token in tokens:
        vocab_counter[token] += 1

    vocab_freq = vocab_counter.most_common(self.num_words)
    vocab_list = [self.PAD, self.START, self.UNKNOWN
                 ] + [word for word, _ in vocab_freq]
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

  compat_tf_versions = _get_compat_tf_versions(2)
  need_gen_vocab = False
  default_batch_size = 32

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
      convert_from_saved_model_tf2=False,
      name='Bert',
      tflite_input_name=None):
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
      convert_from_saved_model_tf2: Convert to TFLite from saved_model in TF
        2.x.
      name: The name of the object.
      tflite_input_name: Dict, input names for the TFLite model.
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

    num_gpus = get_num_gpus(num_gpus)
    self.strategy = distribution_utils.get_distribution_strategy(
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

    self.convert_from_saved_model_tf2 = convert_from_saved_model_tf2
    self.is_built = False
    self.name = name

    if tflite_input_name is None:
      tflite_input_name = {
          'ids': 'input_word_ids',
          'mask': 'input_mask',
          'segment_ids': 'input_type_ids'
      }
    self.tflite_input_name = tflite_input_name

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

  def create_model(self, num_classes, optimizer='adam'):
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

    bert_model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[metric_fn()])

    return bert_model

  def run_classifier(self, train_input_fn, validation_input_fn, epochs,
                     steps_per_epoch, validation_steps, num_classes):
    """Creates classifier and runs the classifier training."""

    warmup_steps = int(epochs * steps_per_epoch * 0.1)
    initial_lr = self.learning_rate

    with distribution_utils.get_strategy_scope(self.strategy):
      training_dataset = train_input_fn()
      evaluation_dataset = None
      if validation_input_fn is not None:
        evaluation_dataset = validation_input_fn()

      optimizer = optimization.create_optimizer(initial_lr,
                                                steps_per_epoch * epochs,
                                                warmup_steps)
      bert_model = self.create_model(num_classes, optimizer)

    summary_dir = os.path.join(self.model_dir, 'summaries')
    summary_callback = tf.keras.callbacks.TensorBoard(summary_dir)
    checkpoint_path = os.path.join(self.model_dir, 'checkpoint')
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path, save_weights_only=True)

    bert_model.fit(
        x=training_dataset,
        validation_data=evaluation_dataset,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_steps=validation_steps,
        callbacks=[summary_callback, checkpoint_callback])

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
      convert_from_saved_model_tf2=False,
      tflite_input_name=None,
      tflite_output_name=None,
      init_from_squad_model=False,
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
      convert_from_saved_model_tf2: Convert to TFLite from saved_model in TF
        2.x.
      tflite_input_name: Dict, input names for the TFLite model.
      tflite_output_name: Dict, output names for the TFLite model.
      init_from_squad_model: boolean, whether to initialize from the model that
        is already retrained on Squad 1.1.
      name: Name of the object.
    """
    super(BertQAModelSpec,
          self).__init__(uri, model_dir, seq_len, dropout_rate,
                         initializer_range, learning_rate,
                         distribution_strategy, num_gpus, tpu, trainable,
                         do_lower_case, is_tf2, convert_from_saved_model_tf2,
                         name, tflite_input_name)
    self.query_len = query_len
    self.doc_stride = doc_stride
    self.predict_batch_size = predict_batch_size
    if tflite_output_name is None:
      tflite_output_name = {
          'start_logits': 'Identity_1',
          'end_logits': 'Identity'
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

  def train(self, train_input_fn, epochs, steps_per_epoch):
    """Run bert QA training."""
    warmup_steps = int(epochs * steps_per_epoch * 0.1)

    def _loss_fn(positions, logits):
      """Get losss function for QA model."""
      loss = tf.keras.losses.sparse_categorical_crossentropy(
          positions, logits, from_logits=True)
      return tf.reduce_mean(loss)

    with distribution_utils.get_strategy_scope(self.strategy):
      training_dataset = train_input_fn()
      bert_model = self.create_model()
      optimizer = optimization.create_optimizer(self.learning_rate,
                                                steps_per_epoch * epochs,
                                                warmup_steps)

      bert_model.compile(
          optimizer=optimizer, loss=_loss_fn, loss_weights=[0.5, 0.5])

    summary_dir = os.path.join(self.model_dir, 'summaries')
    summary_callback = tf.keras.callbacks.TensorBoard(summary_dir)
    checkpoint_path = os.path.join(self.model_dir, 'checkpoint')
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path, save_weights_only=True)

    if not bert_model.trainable_variables:
      tf.compat.v1.logging.warning(
          'Trainable variables in the model are empty.')
      return bert_model

    bert_model.fit(
        x=training_dataset,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=[summary_callback, checkpoint_callback])

    return bert_model

  def _predict_without_distribute_strategy(self, model, input_fn):
    """Predicts the dataset without using distribute strategy."""
    ds = input_fn()
    all_results = []
    for features, _ in ds:
      outputs = model.predict_on_batch(features)
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

  def _predict_with_distribute_strategy(self, model, input_fn, num_steps):
    """Predicts the dataset using distribute strategy."""
    predict_iterator = iter(
        self.strategy.experimental_distribute_datasets_from_function(input_fn))

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

      outputs = self.strategy.run(_replicated_step, args=(next(iterator),))
      return tf.nest.map_structure(self.strategy.experimental_local_results,
                                   outputs)

    all_results = []
    for _ in range(num_steps):
      predictions = predict_step(predict_iterator)
      for result in run_squad_helper.get_raw_results(predictions):
        all_results.append(result)
      if len(all_results) % 100 == 0:
        tf.compat.v1.logging.info('Made predictions for %d records.',
                                  len(all_results))
    return all_results

  def predict(self, model, input_fn, num_steps):
    """Predicts the dataset from `input_fn` for `model`."""
    if self.strategy:
      return self._predict_with_distribute_strategy(model, input_fn, num_steps)
    else:
      return self._predict_without_distribute_strategy(model, input_fn)

  def reorder_output_details(self, tflite_output_details):
    """Reorders the tflite output details to map the order of keras model."""
    for detail in tflite_output_details:
      name = detail['name']
      if self.tflite_output_name['start_logits'] == name:
        start_logits_detail = detail
      if self.tflite_output_name['end_logits'] == name:
        end_logits_detail = detail
    return (start_logits_detail, end_logits_detail)

  def predict_tflite(self, tflite_filepath, input_fn):
    """Predicts the `input_fn` dataset for TFLite model in `tflite_filepath`."""
    ds = input_fn()
    all_results = []
    lite_runner = model_util.LiteRunner(tflite_filepath,
                                        self.reorder_input_details,
                                        self.reorder_output_details)
    for features, _ in ds:
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

  def evaluate(self, model, tflite_filepath, input_fn, num_steps, eval_examples,
               eval_features, predict_file, version_2_with_negative,
               max_answer_length, null_score_diff_threshold, verbose_logging,
               output_dir):
    """Evaluate QA model.

    Args:
      model: The keras model to be evaluated.
      tflite_filepath: File path to the TFLite model.
      input_fn: Function that returns a tf.data.Dataset used for evaluation.
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
      all_results = self.predict_tflite(tflite_filepath, input_fn)
    else:
      all_results = self.predict(model, input_fn, num_steps)

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


_MOBILEBERT_TFLITE_INPUT_NAME = {
    'ids': 'serving_default_input_word_ids:0',
    'mask': 'serving_default_input_mask:0',
    'segment_ids': 'serving_default_input_type_ids:0'
}
_MOBILEBERT_TFLITE_OUTPUT_NAME = {
    'start_logits': 'StatefulPartitionedCall:1',
    'end_logits': 'StatefulPartitionedCall:0'
}

mobilebert_classifier_spec = BertClassifierModelSpec(
    uri='https://tfhub.dev/google/mobilebert/uncased_L-24_H-128_B-512_A-4_F-4_OPT/1',
    is_tf2=False,
    distribution_strategy='off',
    convert_from_saved_model_tf2=True,
    name='MobileBert',
    tflite_input_name=_MOBILEBERT_TFLITE_INPUT_NAME)
mobilebert_classifier_spec.default_batch_size = 48

mobilebert_qa_spec = BertQAModelSpec(
    uri='https://tfhub.dev/google/mobilebert/uncased_L-24_H-128_B-512_A-4_F-4_OPT/1',
    is_tf2=False,
    distribution_strategy='off',
    convert_from_saved_model_tf2=True,
    learning_rate=5e-05,
    name='MobileBert',
    tflite_input_name=_MOBILEBERT_TFLITE_INPUT_NAME,
    tflite_output_name=_MOBILEBERT_TFLITE_OUTPUT_NAME)
mobilebert_qa_spec.default_batch_size = 48

mobilebert_qa_squad_spec = BertQAModelSpec(
    uri='https://tfhub.dev/google/mobilebert/uncased_L-24_H-128_B-512_A-4_F-4_OPT/squadv1/1',
    is_tf2=False,
    distribution_strategy='off',
    convert_from_saved_model_tf2=True,
    learning_rate=5e-05,
    name='MobileBert',
    tflite_input_name=_MOBILEBERT_TFLITE_INPUT_NAME,
    tflite_output_name=_MOBILEBERT_TFLITE_OUTPUT_NAME,
    init_from_squad_model=True)
mobilebert_qa_squad_spec.default_batch_size = 48

# A dict for model specs to make it accessible by string key.
MODEL_SPECS = {
    'efficientnet_lite0': efficientnet_lite0_spec,
    'efficientnet_lite1': efficientnet_lite1_spec,
    'efficientnet_lite2': efficientnet_lite2_spec,
    'efficientnet_lite3': efficientnet_lite3_spec,
    'efficientnet_lite4': efficientnet_lite4_spec,
    'mobilenet_v2': mobilenet_v2_spec,
    'resnet_50': resnet_50_spec,
    'average_word_vec': AverageWordVecModelSpec,
    'bert': BertModelSpec,
    'bert_classifier': BertClassifierModelSpec,
    'bert_qa': BertQAModelSpec,
    'mobilebert_classifier': mobilebert_classifier_spec,
    'mobilebert_qa': mobilebert_qa_spec,
    'mobilebert_qa_squad': mobilebert_qa_squad_spec,
}

# List constants for supported models.
IMAGE_CLASSIFICATION_MODELS = [
    'efficientnet_lite0', 'efficientnet_lite1', 'efficientnet_lite2',
    'efficientnet_lite3', 'efficientnet_lite4', 'mobilenet_v2', 'resnet_50'
]
TEXT_CLASSIFICATION_MODELS = [
    'bert_classifier', 'average_word_vec', 'mobilebert_classifier'
]
QUESTION_ANSWERING_MODELS = ['bert_qa', 'mobilebert_qa', 'mobilebert_qa_squad']


def get(spec_or_str):
  """Gets model spec by name or instance, and initializes by default."""
  if isinstance(spec_or_str, str):
    model_spec = MODEL_SPECS[spec_or_str]
  else:
    model_spec = spec_or_str

  if inspect.isclass(model_spec):
    return model_spec()
  else:
    return model_spec
