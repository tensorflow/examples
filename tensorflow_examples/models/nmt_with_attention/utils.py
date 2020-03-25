# coding=utf-8

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
# ==============================================================================
"""Utils.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import os
import re
import unicodedata
from absl import flags
from sklearn.model_selection import train_test_split

import tensorflow as tf

FLAGS = flags.FLAGS

_URL = 'http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip'


def nmt_flags():
  flags.DEFINE_string('download_path', 'datasets', 'Download folder')
  flags.DEFINE_integer('buffer_size', 70000, 'Shuffle buffer size')
  flags.DEFINE_integer('batch_size', 64, 'Batch Size')
  flags.DEFINE_integer('epochs', 1, 'Number of epochs')
  flags.DEFINE_integer('embedding_dim', 256, 'Embedding dimension')
  flags.DEFINE_integer('enc_units', 1024, 'Encoder GRU units')
  flags.DEFINE_integer('dec_units', 1024, 'Decoder GRU units')
  flags.DEFINE_boolean('enable_function', True, 'Enable Function?')
  flags.DEFINE_integer('num_examples', 70000, 'Number of examples from dataset')


def download(download_path):
  path_to_zip = tf.keras.utils.get_file(
      'spa-eng.zip', origin=_URL, cache_subdir=download_path,
      extract=True)
  path_to_file = os.path.join(os.path.dirname(path_to_zip), 'spa-eng/spa.txt')

  return path_to_file


def unicode_to_ascii(s):
  return ''.join(c for c in unicodedata.normalize('NFD', s)
                 if unicodedata.category(c) != 'Mn')


def preprocess_sentence(w):
  """Preprocessing words in a sentence.

  Args:
    w: Word.

  Returns:
    Preprocessed words.
  """

  w = unicode_to_ascii(w.lower().strip())

  # creating a space between a word and the punctuation following it
  w = re.sub(r'([?.!,¿])', r' \1 ', w)
  w = re.sub(r'[" "]+', ' ', w)

  # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
  w = re.sub(r'[^a-zA-Z?.!,¿]+', ' ', w)

  w = w.rstrip().strip()

  # adding a start and an end token to the sentence
  # so that the model know when to start and stop predicting.
  w = '<start> ' + w + ' <end>'
  return w


def create_word_pairs(path, num_examples):
  lines = io.open(path, encoding='UTF-8').read().strip().split('\n')

  word_pairs = [[preprocess_sentence(w) for w in l.split('\t')]  # pylint: disable=g-complex-comprehension
                for l in lines[:num_examples]]

  return zip(*word_pairs)


def max_length(tensor):
  return max(len(t) for t in tensor)


def tokenize(lang):
  """Tokenize the languages.

  Args:
    lang: Language to be tokenized.

  Returns:
    tensor: Tensors generated after tokenization.
    lang_tokenizer: tokenizer.
  """

  lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(
      filters='')
  lang_tokenizer.fit_on_texts(lang)

  tensor = lang_tokenizer.texts_to_sequences(lang)

  tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                         padding='post')

  return tensor, lang_tokenizer


def load_dataset(path, num_examples):
  # creating cleaned input, output pairs
  targ_lang, inp_lang = create_word_pairs(path, num_examples)

  input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
  target_tensor, targ_lang_tokenizer = tokenize(targ_lang)

  return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer


def create_dataset(path_to_file, num_examples, buffer_size, batch_size):
  """Create a tf.data Dataset.

  Args:
    path_to_file: Path to the file to load the text from.
    num_examples: Number of examples to sample.
    buffer_size: Shuffle buffer size.
    batch_size: Batch size.

  Returns:
    train_dataset: Training dataset.
    test_dataset: Test dataset.
    inp_lang: Input language tokenizer.
    targ_lang: Target language tokenizer.
  """

  input_tensor, target_tensor, inp_lang, targ_lang = load_dataset(
      path_to_file, num_examples)

  # Creating training and validation sets using an 80-20 split
  inp_train, inp_val, target_train, target_val = train_test_split(
      input_tensor, target_tensor, test_size=0.2)

  # Create a tf.data dataset
  train_dataset = tf.data.Dataset.from_tensor_slices(
      (inp_train, target_train)).shuffle(buffer_size)
  train_dataset = train_dataset.batch(batch_size, drop_remainder=True)

  test_dataset = tf.data.Dataset.from_tensor_slices((inp_val, target_val))
  test_dataset = test_dataset.batch(batch_size, drop_remainder=True)

  return train_dataset, test_dataset, inp_lang, targ_lang


def get_common_kwargs():
  return {'epochs': 1, 'enable_function': True, 'buffer_size': 70000,
          'batch_size': 64, 'download_path': 'datasets'}


def flags_dict():
  """Define the flags.

  Returns:
    Command line arguments as Flags.
  """

  kwargs = {
      'epochs': FLAGS.epochs,
      'enable_function': FLAGS.enable_function,
      'buffer_size': FLAGS.buffer_size,
      'batch_size': FLAGS.batch_size,
      'download_path': FLAGS.download_path,
      'num_examples': FLAGS.num_examples,
      'embedding_dim': FLAGS.embedding_dim,
      'enc_units': FLAGS.enc_units,
      'dec_units': FLAGS.dec_units
  }

  return kwargs
