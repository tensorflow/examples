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
    return

  @abc.abstractmethod
  def preprocess(self, raw_text, label):
    """Preprocesses function for the text model."""
    return

  @abc.abstractmethod
  def save_vocab(self, vocab_filename):
    """Saves the vocabulary if it's generated from the input data, otherwise, prints the file path to the vocabulary."""
    return


class AverageWordVecModelSpec(TextModelSpec):
  """A specification of averaging word vector model."""
  PAD = '<PAD>'  # Index: 0
  START = '<START>'  # Index: 1
  UNKNOWN = '<UNKNOWN>'  # Index: 2

  def __init__(self,
               num_words=10000,
               sentence_len=256,
               wordvec_dim=16,
               lowercase=True,
               dropout_rate=0.2):
    """Initialze a instance with preprocessing and model paramaters.

    Args:
      num_words: Number of words to generate the vocabulary from data.
      sentence_len: Length of the sentence to feed into the model.
      wordvec_dim: Dimension of the word embedding.
      lowercase: Whether to convert all uppercase character to lowercase during
        preprocessing.
      dropout_rate: The rate for dropout.
    """
    super(AverageWordVecModelSpec, self).__init__(need_gen_vocab=True)
    self.num_words = num_words
    self.sentence_len = sentence_len
    self.wordvec_dim = wordvec_dim
    self.lowercase = lowercase
    self.dropout_rate = dropout_rate

  def run_classifier(self, train_input_fn, validation_input_fn, epochs,
                     steps_per_epoch, validation_steps, num_classes):
    """Creates classifier and runs the classifier training."""
    # Gets a classifier model.
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(
            len(self.vocab), self.wordvec_dim, input_length=self.sentence_len),
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

  def gen_vocab(self, text_ds):
    """Generates vocabulary list in `text_ds` with maximum `num_words` words."""
    vocab_counter = collections.Counter()

    for raw_text, _ in text_ds:
      tokens = self._tokenize(raw_text.numpy())
      for token in tokens:
        vocab_counter[token] += 1

    vocab_freq = vocab_counter.most_common(self.num_words)
    vocab_list = [self.PAD, self.START, self.UNKNOWN
                 ] + [word for word, _ in vocab_freq]
    self.vocab = collections.OrderedDict(
        ((v, i) for i, v in enumerate(vocab_list)))
    return self.vocab

  def _preprocess_text_py_func(self, raw_text, label):
    """Preprocess the text."""
    tokens = self._tokenize(raw_text.numpy())

    # Gets ids for START, PAD and UNKNOWN tokens.
    start_id = self.vocab[self.START]
    pad_id = self.vocab[self.PAD]
    unknown_id = self.vocab[self.UNKNOWN]

    token_ids = [self.vocab.get(token, unknown_id) for token in tokens]
    token_ids = [start_id] + token_ids

    if len(token_ids) < self.sentence_len:
      # Padding.
      pad_length = self.sentence_len - len(token_ids)
      token_ids = token_ids + pad_length * [pad_id]
    else:
      token_ids = token_ids[:self.sentence_len]
    return token_ids, label

  def preprocess(self, raw_text, label):
    """Preprocess the text."""
    text, label = tf.py_function(
        self._preprocess_text_py_func,
        inp=[raw_text, label],
        Tout=(tf.int32, tf.int64))
    text.set_shape((self.sentence_len))
    label.set_shape(())
    return text, label

  def _tokenize(self, sentence):
    r"""Splits by '\W' except '\''."""
    sentence = tf.compat.as_text(sentence)
    if self.lowercase:
      sentence = sentence.lower()
    tokens = re.compile(r'[^\w\']+').split(sentence.strip())
    return list(filter(None, tokens))

  def save_vocab(self, vocab_filename):
    """Saves the vocabulary in `vocab_filename`."""
    with tf.io.gfile.GFile(vocab_filename, 'w') as f:
      for token, index in self.vocab.items():
        f.write('%s %d\n' % (token, index))

    tf.compat.v1.logging.info('Saved vocabulary in %s.', vocab_filename)
