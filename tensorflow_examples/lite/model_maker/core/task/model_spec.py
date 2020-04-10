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

import tensorflow.compat.v2 as tf
from tensorflow_examples.lite.model_maker.core.task import hub_loader

import tensorflow_hub as hub
from tensorflow_hub import registry

from official.nlp import optimization
from official.nlp.bert import configs as bert_configs
from official.nlp.bert import tokenization
from official.nlp.data import classifier_data_lib
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
  tot_num_gpus = len(tf.config.experimental.list_physical_devices('GPU'))
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

  def __init__(self,
               num_words=10000,
               seq_len=256,
               wordvec_dim=16,
               lowercase=True,
               dropout_rate=0.2):
    """Initialze a instance with preprocessing and model paramaters.

    Args:
      num_words: Number of words to generate the vocabulary from data.
      seq_len: Length of the sequence to feed into the model.
      wordvec_dim: Dimension of the word embedding.
      lowercase: Whether to convert all uppercase character to lowercase during
        preprocessing.
      dropout_rate: The rate for dropout.
    """
    self.num_words = num_words
    self.seq_len = seq_len
    self.wordvec_dim = wordvec_dim
    self.lowercase = lowercase
    self.dropout_rate = dropout_rate

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

  def run_classifier(self, train_input_fn, validation_input_fn, epochs,
                     steps_per_epoch, validation_steps, num_classes):
    """Creates classifier and runs the classifier training."""
    if epochs is None:
      epochs = self.default_training_epochs

    # Gets a classifier model.
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=[self.seq_len]),
        tf.keras.layers.Embedding(
            len(self.vocab), self.wordvec_dim, input_length=self.seq_len),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(self.wordvec_dim, activation=tf.nn.relu),
        tf.keras.layers.Dropout(self.dropout_rate),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    # Gets training and validation dataset
    train_ds = train_input_fn()
    validation_ds = None
    if validation_input_fn is not None:
      validation_ds = validation_input_fn()

    # Trains the models.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

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
      dtype=tf.float32)(
          output)

  return tf.keras.Model(
      inputs=[input_word_ids, input_mask, input_type_ids],
      outputs=output), bert_model


class BertModelSpec(object):
  """A specification of BERT model."""

  compat_tf_versions = _get_compat_tf_versions(2)
  need_gen_vocab = False
  default_training_epochs = 3

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
      is_tf2=True):
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
    """
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

    self.uri = uri

    self.is_tf2 = is_tf2
    self.vocab_file = os.path.join(
        registry.resolver(uri), 'assets', 'vocab.txt')
    self.do_lower_case = do_lower_case

    self.tokenizer = tokenization.FullTokenizer(self.vocab_file,
                                                self.do_lower_case)

    self.bert_config = bert_configs.BertConfig(
        0,
        initializer_range=self.initializer_range,
        hidden_dropout_prob=self.dropout_rate)

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
    classifier_data_lib.file_based_convert_examples_to_features(
        examples, label_names, self.seq_len, self.tokenizer, tfrecord_file)

  def _get_classification_loss_fn(self, num_classes):
    """Gets the classification loss function."""

    def _classification_loss_fn(labels, logits):
      """Classification loss."""
      labels = tf.squeeze(labels)
      log_probs = tf.nn.log_softmax(logits, axis=-1)
      one_hot_labels = tf.one_hot(
          tf.cast(labels, dtype=tf.int32), depth=num_classes, dtype=tf.float32)
      per_example_loss = -tf.reduce_sum(
          tf.cast(one_hot_labels, dtype=tf.float32) * log_probs, axis=-1)
      loss = tf.reduce_mean(per_example_loss)
      return loss

    return _classification_loss_fn

  def run_classifier(self, train_input_fn, validation_input_fn, epochs,
                     steps_per_epoch, validation_steps, num_classes):
    """Creates classifier and runs the classifier training."""
    if epochs is None:
      epochs = self.default_training_epochs

    warmup_steps = int(epochs * steps_per_epoch * 0.1)
    initial_lr = self.learning_rate

    def _get_classifier_model():
      """Gets a classifier model."""
      classifier_model, core_model = create_classifier_model(
          self.bert_config,
          num_classes,
          self.seq_len,
          hub_module_url=self.uri,
          hub_module_trainable=self.trainable,
          is_tf2=self.is_tf2)

      classifier_model.optimizer = optimization.create_optimizer(
          initial_lr, steps_per_epoch * epochs, warmup_steps)
      return classifier_model, core_model

    loss_fn = self._get_classification_loss_fn(num_classes)

    # Defines evaluation metrics function, which will create metrics in the
    # correct device and strategy scope.
    def metric_fn():
      return tf.keras.metrics.SparseCategoricalAccuracy(
          'test_accuracy', dtype=tf.float32)

    with distribution_utils.get_strategy_scope(self.strategy):
      training_dataset = train_input_fn()
      evaluation_dataset = None
      if validation_input_fn is not None:
        evaluation_dataset = validation_input_fn()
      bert_model, _ = _get_classifier_model()
      optimizer = bert_model.optimizer

      bert_model.compile(
          optimizer=optimizer, loss=loss_fn, metrics=[metric_fn()])

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

  def save_vocab(self, vocab_filename):
    """Prints the file path to the vocabulary."""
    tf.io.gfile.copy(self.vocab_file, vocab_filename, overwrite=True)
    tf.compat.v1.logging.info('Saved vocabulary in %s.', vocab_filename)

  def get_config(self):
    """Gets the configuration."""
    # Only preprocessing related variables are included.
    return {'uri': self.uri, 'seq_len': self.seq_len}


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
}

# List constants for supported models.
IMAGE_CLASSIFICATION_MODELS = [
    'efficientnet_lite0', 'efficientnet_lite1', 'efficientnet_lite2',
    'efficientnet_lite3', 'efficientnet_lite4', 'mobilenet_v2', 'resnet_50'
]
TEXT_CLASSIFICATION_MODELS = ['bert', 'average_word_vec']


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
