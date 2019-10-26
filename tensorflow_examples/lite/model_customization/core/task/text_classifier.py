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
import os
import re
import tempfile

import numpy as np
import tensorflow as tf # TF2
import tensorflow_examples.lite.model_customization.core.model_export_format as mef
from tensorflow_examples.lite.model_customization.core.task import classification_model


PAD = '<PAD>'  # Index: 0
START = '<START>'  # Index: 1
UNKNOWN = '<UNKNOWN>'  # Index: 2


def create(data,
           model_export_format=mef.ModelExportFormat.TFLITE,
           model_name='average_wordvec',
           shuffle=False,
           batch_size=32,
           epochs=2,
           validation_ratio=0.1,
           test_ratio=0.1,
           num_words=10000,
           sentence_len=256,
           wordvec_dim=16,
           dropout_rate=0.2,
           lowercase=True):
  """Loads data and train the model for test classification.

  Args:
    data: Raw data that could be splitted for training / validation / testing.
    model_export_format: Model export format such as saved_model / tflite.
    model_name: Model name.
    shuffle: Whether the data should be shuffled.
    batch_size: Batch size for training.
    epochs: Number of epochs for training.
    validation_ratio: The ratio of valid data to be splitted.
    test_ratio: The ratio of test data to be splitted.
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
      data,
      model_export_format,
      model_name,
      shuffle=shuffle,
      validation_ratio=validation_ratio,
      test_ratio=test_ratio,
      num_words=num_words,
      sentence_len=sentence_len,
      wordvec_dim=wordvec_dim,
      dropout_rate=dropout_rate,
      lowercase=lowercase)

  tf.compat.v1.logging.info('Retraining the models...')
  text_classifier.train(epochs, batch_size)

  return text_classifier


class TextClassifier(classification_model.ClassificationModel):
  """TextClassifier class for inference and exporting to tflite."""

  def __init__(self,
               data,
               model_export_format,
               model_name,
               shuffle=True,
               validation_ratio=0.1,
               test_ratio=0.1,
               num_words=10000,
               sentence_len=256,
               wordvec_dim=16,
               dropout_rate=0.2,
               lowercase=True):
    """Init function for TextClassifier class.

    Including splitting the raw input data into train/eval/test sets and
    selecting the exact NN model to be used.

    Args:
      data: Raw data that could be splitted for training / validation / testing.
      model_export_format: Model export format such as saved_model / tflite.
      model_name: Model name.
      shuffle: Whether the data should be shuffled.
      validation_ratio: The ratio of valid data to be splitted.
      test_ratio: The ratio of test data to be splitted.
      num_words: Number of words to generate the vocabulary from data.
      sentence_len: Length of the sentence to feed into the model.
      wordvec_dim: Dimension of the word embedding.
      dropout_rate: the rate for dropout.
      lowercase: Whether to convert all uppercase character to lowercase
        during preprocessing.
    """
    if model_export_format != mef.ModelExportFormat.TFLITE:
      raise ValueError('Model export format %s is not supported currently.' %
                       str(model_export_format))

    # Checks model parameter.
    if model_name != 'average_wordvec':
      raise ValueError('Model %s is not supported currently.' % model_name)

    super(TextClassifier, self).__init__(
        data,
        model_export_format,
        model_name,
        shuffle,
        train_whole_model=False,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio)
    self.sentence_len = sentence_len
    self.lowercase = lowercase

    self.vocab = self._gen_vocab(data.dataset, num_words)

    # Generates training, validation and testing data.
    if validation_ratio + test_ratio >= 1.0:
      raise ValueError(
          'The total ratio for validation and test data should be less than 1.0.'
      )

    self.valid_data, rest_data = data.split(validation_ratio, shuffle=shuffle)
    self.test_data, self.train_data = rest_data.split(
        test_ratio, shuffle=shuffle)

    # Checks dataset parameter.
    if self.train_data.size == 0:
      raise ValueError('Training dataset is empty.')

    # Creates the text classifier model.
    self.model = tf.keras.Sequential([
        tf.keras.layers.Embedding(
            len(self.vocab), wordvec_dim, input_length=sentence_len),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(wordvec_dim, activation=tf.nn.relu),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(self.data.num_classes, activation='softmax')
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

  def train(self, epochs, batch_size=32):
    """Feeds the training data for training."""

    train_ds = self._gen_train_dataset(self.train_data, batch_size)
    valid_ds = self._gen_valid_dataset(self.valid_data, batch_size)

    # Trains the models.
    self.model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

    steps_per_epoch = self.train_data.size // batch_size
    validation_steps = self.valid_data.size // batch_size
    self.model.fit(
        train_ds,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=valid_ds,
        validation_steps=validation_steps)

  def evaluate(self, data=None, batch_size=32):
    """Evaluates the model.

    Args:
      data: Data to be evaluated. If None, then evaluates in self.test_data.
      batch_size: Number of samples per evaluation step.

    Returns:
      The loss value and accuracy.
    """
    if data is None:
      data = self.test_data
    ds = self._gen_valid_dataset(data, batch_size)

    return self.model.evaluate(ds)

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

  def preprocess_text(self, raw_text, label):
    text, label = tf.py_function(
        self._preprocess_text_py_func,
        inp=[raw_text, label],
        Tout=(tf.int32, tf.int64))
    text.set_shape((self.sentence_len))
    label.set_shape(())
    return text, label

  def _gen_train_dataset(self, data, batch_size=32):
    ds = data.dataset.map(self.preprocess_text)
    if self.shuffle:
      ds = ds.shuffle(buffer_size=data.size)
    ds = ds.repeat()
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds

  def _gen_valid_dataset(self, data, batch_size=32):
    ds = data.dataset.map(self.preprocess_text)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds

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
    temp_dir = tempfile.TemporaryDirectory()
    kera_model_file = os.path.join(temp_dir.name, 'tmp.h5')
    self.model.save(kera_model_file)

    input_embedding = '{}_input'.format(self.model.layers[0].name)
    converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file(
        kera_model_file,
        input_shapes={input_embedding: [None, self.sentence_len]})
    if quantized:
      converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    tflite_model = converter.convert()

    with tf.io.gfile.GFile(tflite_filename, 'wb') as f:
      f.write(tflite_model)

    with tf.io.gfile.GFile(label_filename, 'w') as f:
      f.write('\n'.join(self.data.index_to_label))

    with tf.io.gfile.GFile(vocab_filename, 'w') as f:
      for token, index in self.vocab.items():
        f.write('%s %d\n' % (token, index))

    tf.compat.v1.logging.info(
        'Exported to tflite model in %s, saved labels in %s and vocabulary in %s.',
        tflite_filename, label_filename, vocab_filename)

  # TODO(b/142607208): need to fix the wrong output.
  def predict_topk(self, data=None, k=1, batch_size=32):
    """Predicts the top-k predictions.

    Args:
      data: Data to be evaluated. If None, then predicts in self.test_data.
      k: Number of top results to be predicted.
      batch_size: Number of samples per evaluation step.

    Returns:
      top k results. Each one is (label, probability).
    """
    if k < 0:
      raise ValueError('K should be equal or larger than 0.')

    if data is None:
      data = self.test_data
    ds = self._gen_valid_dataset(data, batch_size)

    predicted_prob = self.model.predict(ds)
    topk_prob, topk_id = tf.math.top_k(predicted_prob, k=k)
    topk_label = np.array(self.data.index_to_label)[topk_id]

    label_prob = []
    for label, prob in zip(topk_label, topk_prob.numpy()):
      label_prob.append(list(zip(label, prob)))

    return label_prob
