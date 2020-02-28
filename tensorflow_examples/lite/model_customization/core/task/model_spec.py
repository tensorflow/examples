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

import abc
import collections
import re
import tempfile

import tensorflow as tf # TF2
import tensorflow_hub as hub

from official.modeling import model_training_utils
from official.nlp import optimization
from official.nlp.bert import bert_models
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

  input_image_shape = [224, 224]
  mean_rgb = [0, 0, 0]
  stddev_rgb = [255, 255, 255]

  def __init__(self, uri, compat_tf_versions=None):
    self.uri = uri
    self.compat_tf_versions = _get_compat_tf_versions(compat_tf_versions)


efficientnet_b0_spec = ImageModelSpec(
    uri='https://tfhub.dev/google/efficientnet/b0/feature-vector/1',
    compat_tf_versions=[1, 2])

mobilenet_v2_spec = ImageModelSpec(
    uri='https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4',
    compat_tf_versions=2)

resnet_50_spec = ImageModelSpec(
    uri='https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/4',
    compat_tf_versions=2)


class TextModelSpec(abc.ABC):
  """The abstract base class that constains the specification of text model."""

  def __init__(self, need_gen_vocab, experimental_new_converter=False):
    """Initialization function for TextClassifier class.

    Args:
      need_gen_vocab: If true, needs to generate vocabulary from input data
        using `gen_vocab` function. Otherwise, loads vocab from text model
        assets.
      experimental_new_converter: Experimental flag, subject to change. Enables
        MLIR-based conversion instead of TOCO conversion.
    """
    self.need_gen_vocab = need_gen_vocab
    self.experimental_new_converter = experimental_new_converter

  @abc.abstractmethod
  def run_classifier(self, train_input_fn, validation_input_fn, epochs,
                     steps_per_epoch, validation_steps, num_classes):
    """Creates classifier and runs the classifier training.

    Args:
      train_input_fn: Function that returns a tf.data.Dataset used for training.
      validation_input_fn: Function that returns a tf.data.Dataset used for
        evaluation.
      epochs: Number of epochs to train.
      steps_per_epoch: Number of steps to run per epoch.
      validation_steps: Number of steps to run evaluation.
      num_classes: Number of label classes.

    Returns:
      The retrained model.
    """
    pass

  @abc.abstractmethod
  def save_vocab(self, vocab_filename):
    """Saves the vocabulary if it's generated from the input data, otherwise, prints the file path to the vocabulary."""
    pass

  @abc.abstractmethod
  def get_name_to_features(self):
    """Gets the dictionary describing the features."""
    pass

  @abc.abstractmethod
  def select_data_from_record(self, record):
    """Dispatches records to features and labels."""
    pass

  @abc.abstractmethod
  def convert_examples_to_features(self, examples, tfrecord_file, label_names):
    """Converts examples to features and write them into TFRecord file."""
    pass

  @abc.abstractmethod
  def get_config(self):
    """Gets the configuration."""
    pass

  def set_shape(self, model):
    """Sets the input model shape. Used in tflite conveter for BatchMatMul."""
    pass


class AverageWordVecModelSpec(TextModelSpec):
  """A specification of averaging word vector model."""
  PAD = '<PAD>'  # Index: 0
  START = '<START>'  # Index: 1
  UNKNOWN = '<UNKNOWN>'  # Index: 2

  compat_tf_versions = _get_compat_tf_versions(2)

  def __init__(self,
               num_words=10000,
               seq_len=256,
               wordvec_dim=16,
               lowercase=True,
               dropout_rate=0.2,
               experimental_new_converter=False):
    """Initialze a instance with preprocessing and model paramaters.

    Args:
      num_words: Number of words to generate the vocabulary from data.
      seq_len: Length of the sequence to feed into the model.
      wordvec_dim: Dimension of the word embedding.
      lowercase: Whether to convert all uppercase character to lowercase during
        preprocessing.
      dropout_rate: The rate for dropout.
      experimental_new_converter: Experimental flag, subject to change. Enables
        MLIR-based conversion instead of TOCO conversion.
    """
    super(AverageWordVecModelSpec, self).__init__(
        need_gen_vocab=True,
        experimental_new_converter=experimental_new_converter)
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


class BertModelSpec(TextModelSpec):
  """A specification of BERT model."""

  compat_tf_versions = _get_compat_tf_versions(2)

  def __init__(
      self,
      uri='https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1',
      model_dir=None,
      seq_len=128,
      dropout_rate=0.1,
      initializer_range=0.02,
      learning_rate=3e-5,
      scale_loss=False,
      steps_per_loop=1000,
      distribution_strategy='mirrored',
      num_gpus=-1,
      tpu='',
      experimental_new_converter=True,
  ):
    """Initialze an instance with model paramaters.

    Args:
      uri: TF-Hub path/url to Bert module.
      model_dir: The location of the model checkpoint files.
      seq_len: Length of the sequence to feed into the model.
      dropout_rate: The rate for dropout.
      initializer_range: The stdev of the truncated_normal_initializer for
        initializing all weight matrices.
      learning_rate: The initial learning rate for Adam.
      scale_loss: Whether to divide the loss by number of replica inside the
        per-replica loss function.
      steps_per_loop: Number of steps per graph-mode loop. In order to reduce
        communication in eager context, training logs are printed every
        steps_per_loop.
      distribution_strategy:  A string specifying which distribution strategy to
        use. Accepted values are 'off', 'one_device', 'mirrored',
        'parameter_server', 'multi_worker_mirrored', and 'tpu' -- case
        insensitive. 'off' means not to use Distribution Strategy; 'tpu' means
        to use TPUStrategy using `tpu_address`.
      num_gpus: How many GPUs to use at each worker with the
        DistributionStrategies API. The default is -1, which means utilize all
        available GPUs.
      tpu: TPU address to connect to.
      experimental_new_converter: Experimental flag, subject to change. Enables
        MLIR-based conversion instead of TOCO conversion.
    """
    super(BertModelSpec, self).__init__(
        need_gen_vocab=False,
        experimental_new_converter=experimental_new_converter)
    self.seq_len = seq_len
    self.dropout_rate = dropout_rate
    self.initializer_range = initializer_range
    self.learning_rate = learning_rate
    self.scale_loss = scale_loss
    self.steps_per_loop = steps_per_loop

    self.model_dir = model_dir
    if self.model_dir is None:
      self.model_dir = tempfile.mkdtemp()

    num_gpus = get_num_gpus(num_gpus)
    self.strategy = distribution_utils.get_distribution_strategy(
        distribution_strategy=distribution_strategy,
        num_gpus=num_gpus,
        tpu_address=tpu)

    self.uri = uri
    self.bert_model = hub.KerasLayer(uri, trainable=True)
    self.vocab_file = self.bert_model.resolved_object.vocab_file.asset_path.numpy(
    )
    self.do_lower_case = self.bert_model.resolved_object.do_lower_case.numpy()

    self.tokenizer = tokenization.FullTokenizer(self.vocab_file,
                                                self.do_lower_case)

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

  def get_classification_loss_fn(self, num_classes, loss_factor=1.0):
    """Gets the classification loss function."""

    def classification_loss_fn(labels, logits):
      """Classification loss."""
      labels = tf.squeeze(labels)
      log_probs = tf.nn.log_softmax(logits, axis=-1)
      one_hot_labels = tf.one_hot(
          tf.cast(labels, dtype=tf.int32), depth=num_classes, dtype=tf.float32)
      per_example_loss = -tf.reduce_sum(
          tf.cast(one_hot_labels, dtype=tf.float32) * log_probs, axis=-1)
      loss = tf.reduce_mean(per_example_loss)
      loss *= loss_factor
      return loss

    return classification_loss_fn

  def run_classifier(self, train_input_fn, validation_input_fn, epochs,
                     steps_per_epoch, validation_steps, num_classes):
    """Creates classifier and runs the classifier training."""

    bert_config = bert_configs.BertConfig(
        0,
        initializer_range=self.initializer_range,
        hidden_dropout_prob=self.dropout_rate)
    warmup_steps = int(epochs * steps_per_epoch * 0.1)
    initial_lr = self.learning_rate

    def _get_classifier_model():
      """Gets a classifier model."""
      classifier_model, core_model = (
          bert_models.classifier_model(
              bert_config, num_classes, self.seq_len, hub_module_url=self.uri))
      classifier_model.optimizer = optimization.create_optimizer(
          initial_lr, steps_per_epoch * epochs, warmup_steps)
      return classifier_model, core_model

    # During distributed training, loss used for gradient computation is
    # summed over from all replicas. When Keras compile/fit() API is used,
    # the fit() API internally normalizes the loss by dividing the loss by
    # the number of replicas used for computation. However, when custom
    # training loop is used this is not done automatically and should be
    # done manually by the end user.
    loss_multiplier = 1.0
    if self.scale_loss:
      loss_multiplier = 1.0 / self.strategy.num_replicas_in_sync

    loss_fn = self.get_classification_loss_fn(
        num_classes, loss_factor=loss_multiplier)

    # Defines evaluation metrics function, which will create metrics in the
    # correct device and strategy scope.
    def metric_fn():
      return tf.keras.metrics.SparseCategoricalAccuracy(
          'test_accuracy', dtype=tf.float32)

    # Use user-defined loop to start training.
    tf.compat.v1.logging.info('Training using customized training loop TF 2.0 '
                              'with distribution strategy.')
    bert_model = model_training_utils.run_customized_training_loop(
        strategy=self.strategy,
        model_fn=_get_classifier_model,
        loss_fn=loss_fn,
        model_dir=self.model_dir,
        steps_per_epoch=steps_per_epoch,
        steps_per_loop=self.steps_per_loop,
        epochs=epochs,
        train_input_fn=train_input_fn,
        eval_input_fn=validation_input_fn,
        eval_steps=validation_steps,
        init_checkpoint=None,
        metric_fn=metric_fn,
        custom_callbacks=None,
        run_eagerly=False)

    # Used in evaluation.
    with self.strategy.scope():
      bert_model, _ = _get_classifier_model()
      checkpoint_path = tf.train.latest_checkpoint(self.model_dir)
      checkpoint = tf.train.Checkpoint(model=bert_model)
      checkpoint.restore(checkpoint_path).expect_partial()
      bert_model.compile(loss=loss_fn, metrics=[metric_fn()])
    return bert_model

  def save_vocab(self, vocab_filename):
    """Prints the file path to the vocabulary."""
    tf.io.gfile.copy(self.vocab_file, vocab_filename, overwrite=True)
    tf.compat.v1.logging.info('Saved vocabulary in %s.', vocab_filename)

  def get_config(self):
    """Gets the configuration."""
    # Only preprocessing related variables are included.
    return {'uri': self.uri, 'seq_len': self.seq_len}

  def set_shape(self, model):
    """Sets the input model shape. Used in tflite conveter for BatchMatMul."""
    for model_input in model.inputs:
      model_input.set_shape((1, self.seq_len))
