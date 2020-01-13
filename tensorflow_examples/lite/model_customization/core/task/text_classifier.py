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
"""TextClassier class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import re

import tensorflow as tf # TF2
import tensorflow_examples.lite.model_customization.core.model_export_format as mef
from tensorflow_examples.lite.model_customization.core.task import classification_model


PAD = '<PAD>'  # Index: 0
START = '<START>'  # Index: 1
UNKNOWN = '<UNKNOWN>'  # Index: 2


def create(train_data,
           model_export_format=mef.ModelExportFormat.TFLITE,
           model_name='average_wordvec',
           shuffle=False,
           batch_size=32,
           epochs=2,
           validation_data=None,
           num_words=10000,
           sentence_len=256,
           wordvec_dim=16,
           dropout_rate=0.2,
           lowercase=True):
  """Loads data and train the model for test classification.

  Args:
    train_data: Raw data for training.
    model_export_format: Model export format such as saved_model / tflite.
    model_name: Model name.
    shuffle: Whether the data should be shuffled.
    batch_size: Batch size for training.
    epochs: Number of epochs for training.
    validation_data: Validation data. If None, skips validation process.
    num_words: Number of words to generate the vocabulary from data.
    sentence_len: Length of the sentence to feed into the model.
    wordvec_dim: Dimension of the word embedding.
    dropout_rate: the rate for dropout.
    lowercase: Whether to convert all uppercase character to lowercase during
      preprocessing.

  Returns:
    TextClassifier
  """
  text_classifier = TextClassifier(
      train_data,
      model_export_format,
      model_name,
      train_data.index_to_label,
      train_data.num_classes,
      shuffle=shuffle,
      num_words=num_words,
      sentence_len=sentence_len,
      wordvec_dim=wordvec_dim,
      dropout_rate=dropout_rate,
      lowercase=lowercase)

  tf.compat.v1.logging.info('Retraining the models...')
  text_classifier.train(train_data, validation_data, epochs, batch_size)

  return text_classifier


class TextClassifier(classification_model.ClassificationModel):
  """TextClassifier class for inference and exporting to tflite."""

  def __init__(self,
               train_data,
               model_export_format,
               model_name,
               index_to_label,
               num_classes,
               shuffle=True,
               validation_data=None,
               num_words=10000,
               sentence_len=256,
               wordvec_dim=16,
               dropout_rate=0.2,
               lowercase=True):
    """Init function for TextClassifier class.

    Args:
      train_data: Raw data for training.
      model_export_format: Model export format such as saved_model / tflite.
      model_name: Model name.
      index_to_label: A list that map from index to label class name.
      num_classes: Number of label classes.
      shuffle: Whether the data should be shuffled.
      validation_data: Validation data. If None, skips validation process.
      num_words: Number of words to generate the vocabulary from data.
      sentence_len: Length of the sentence to feed into the model.
      wordvec_dim: Dimension of the word embedding.
      dropout_rate: the rate for dropout.
      lowercase: Whether to convert all uppercase character to lowercase
        during preprocessing.
    """
    # Checks model parameter.
    if model_name != 'average_wordvec':
      raise ValueError('Model %s is not supported currently.' % model_name)

    super(TextClassifier, self).__init__(
        model_export_format,
        model_name,
        index_to_label,
        num_classes,
        shuffle,
        train_whole_model=False)
    self.sentence_len = sentence_len
    self.lowercase = lowercase
    self.vocab = self._gen_vocab(train_data.dataset, num_words)
    self.model = self._create_model(wordvec_dim, sentence_len, dropout_rate)

  def _create_model(self, wordvec_dim, sentence_len, dropout_rate):
    """Creates the text classifier model."""
    return tf.keras.Sequential([
        tf.keras.layers.Embedding(
            len(self.vocab), wordvec_dim, input_length=sentence_len),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(wordvec_dim, activation=tf.nn.relu),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(self.num_classes, activation='softmax')
    ])

  def _gen_vocab(self, text_ds, num_words):
    """Generates vocabulary list in `text_ds` with maximum `num_words` words."""
    vocab_counter = collections.Counter()

    for raw_text, _ in text_ds:
      tokens = self._tokenize(raw_text.numpy())
      for token in tokens:
        vocab_counter[token] += 1

    vocab_freq = vocab_counter.most_common(num_words)
    vocab_list = [PAD, START, UNKNOWN] + [word for word, _ in vocab_freq]
    vocab = collections.OrderedDict(((v, i) for i, v in enumerate(vocab_list)))
    return vocab

  def train(self, train_data, validation_data=None, epochs=2, batch_size=32):
    """Feeds the training data for training."""

    train_ds = self._gen_train_dataset(train_data, batch_size)
    steps_per_epoch = train_data.size // batch_size

    validation_ds = None
    validation_steps = None
    if validation_data is not None:
      validation_ds = self._gen_validation_dataset(validation_data, batch_size)
      validation_steps = validation_data.size // batch_size

    # Trains the models.
    self.model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

    self.model.fit(
        train_ds,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=validation_ds,
        validation_steps=validation_steps)

  def _tokenize(self, sentence):
    r"""Splits by '\W' except '\''."""
    sentence = tf.compat.as_text(sentence)
    if self.lowercase:
      sentence = sentence.lower()
    tokens = re.compile(r'[^\w\']+').split(sentence.strip())
    return list(filter(None, tokens))

  def _preprocess_text_py_func(self, raw_text, label):
    tokens = self._tokenize(raw_text.numpy())

    # Gets ids for START, PAD and UNKNOWN tokens.
    start_id = self.vocab[START]
    pad_id = self.vocab[PAD]
    unknown_id = self.vocab[UNKNOWN]

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
    text, label = tf.py_function(
        self._preprocess_text_py_func,
        inp=[raw_text, label],
        Tout=(tf.int32, tf.int64))
    text.set_shape((self.sentence_len))
    label.set_shape(())
    return text, label

  def export(self, tflite_filename, label_filename, vocab_filename, **kwargs):
    """Converts the retrained model based on `model_export_format`.

    Args:
      tflite_filename: File name to save tflite model.
      label_filename: File name to save labels.
      vocab_filename: File name to save vocabulary.
      **kwargs: Other parameters like `quantized` for tflite model.
    """
    if self.model_export_format == mef.ModelExportFormat.TFLITE:
      if 'quantized' in kwargs:
        quantized = kwargs['quantized']
      else:
        quantized = False
      self._export_tflite(tflite_filename, label_filename, vocab_filename,
                          quantized)
    else:
      raise ValueError('Model export format %s is not supported currently.' %
                       str(self.model_export_format))

  def _export_tflite(self,
                     tflite_filename,
                     label_filename,
                     vocab_filename,
                     quantized=False):
    """Converts the retrained model to tflite format and saves it.

    Args:
      tflite_filename: File name to save tflite model.
      label_filename: File name to save labels.
      vocab_filename: File name to save vocabulary.
      quantized: boolean, if True, save quantized model.
    """
    super(TextClassifier, self)._export_tflite(
        tflite_filename, label_filename, quantized=quantized)

    with tf.io.gfile.GFile(vocab_filename, 'w') as f:
      for token, index in self.vocab.items():
        f.write('%s %d\n' % (token, index))

    tf.compat.v1.logging.info('  Saved vocabulary in %s.', vocab_filename)
