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

import tensorflow as tf # TF2


def create_int_feature(values):
  feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
  return feature


class ImageModelSpec(object):
  """A specification of image model."""

  input_image_shape = [224, 224]
  mean_rgb = [0, 0, 0]
  stddev_rgb = [255, 255, 255]

  def __init__(self, uri):
    self.uri = uri

efficientnet_b0_spec = ImageModelSpec(
    uri='https://tfhub.dev/google/efficientnet/b0/feature-vector/1')

mobilenet_v2_spec = ImageModelSpec(
    uri='https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4')


class TextModelSpec(abc.ABC):
  """The abstract base class that constains the specification of text model."""

  def __init__(self, need_gen_vocab):
    """Initialization function for TextClassifier class.

    Args:
      need_gen_vocab: If true, needs to generate vocabulary from input data
        using `gen_vocab` function. Otherwise, loads vocab from text model
        assets.
    """
    self.need_gen_vocab = need_gen_vocab

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


class AverageWordVecModelSpec(TextModelSpec):
  """A specification of averaging word vector model."""
  PAD = '<PAD>'  # Index: 0
  START = '<START>'  # Index: 1
  UNKNOWN = '<UNKNOWN>'  # Index: 2

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
    super(AverageWordVecModelSpec, self).__init__(need_gen_vocab=True)
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
