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
"""Evaluate.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
from absl import app, flags
import tensorflow as tf # TF2
from tensorflow_examples.models.nmt_with_attention import nmt
from tensorflow_examples.models.nmt_with_attention import utils
assert tf.__version__.startswith('2')

FLAGS = flags.FLAGS

class Evaluate(object):
  """Train class.

  Attributes:
    encoder: Encoder.
    decoder: Decoder.
    inp_lang: Input language tokenizer.
    targ_lang: Target language tokenizer.
    max_length_targ: target language max embedding feature dimension
    max_length_inp: input language max embedding feature dimension
    optimizer: Optimizer.
    loss_object: Object of the loss class.
  """

  def __init__(self, encoder, decoder, inp_lang,
               targ_lang, max_length_targ, max_length_inp):
    self.encoder = encoder
    self.decoder = decoder
    self.inp_lang = inp_lang
    self.targ_lang = targ_lang
    self.optimizer = tf.keras.optimizers.Adam()
    self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
    self.max_length_targ = max_length_targ
    self.max_length_inp = max_length_inp

  def evaluate(self, sentence, beam_width=4):
    """Custom evaluating step.

    Args:
      sentence: input sentence to evaluate.
      beam_width: beam search K.

    Returns:
      result, sentence
    """
    sentence = utils.preprocess_sentence(sentence)
    inputs = [self.inp_lang.word_index[i] for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.\
                        sequence.pad_sequences([inputs],
                                               maxlen=self.max_length_inp,
                                               padding='post')
    inputs = tf.convert_to_tensor(inputs)
    result = ''
    hidden = [tf.zeros((1, self.encoder.enc_units))]
    enc_out, enc_hidden = self.encoder(inputs, hidden)
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([self.targ_lang.word_index['<start>']], 0)

    for _ in range(self.max_length_targ):
      predictions, dec_hidden, _ = self.decoder(dec_input,
                                                dec_hidden,
                                                enc_out)

      predicted_id = tf.argmax(predictions[0]).numpy()
      result += self.targ_lang.index_word[predicted_id] + ' '
      if self.targ_lang.index_word[predicted_id] == '<end>':
        # return result, sentence, attention_plot
        return result, sentence
      # the predicted ID is fed back into the model
      dec_input = tf.expand_dims([predicted_id], 0)
    return result, sentence

def evaluate_flags():
  flags.DEFINE_string('file_path', '/root/.keras/datasets/spa-eng/spa.txt',
                      'Download training dataset file')
  flags.DEFINE_integer('embedding_dim', 256, 'Embedding dimension')
  flags.DEFINE_integer('enc_units', 1024, 'Encoder GRU units')
  flags.DEFINE_integer('dec_units', 1024, 'Decoder GRU units')
  flags.DEFINE_integer('num_examples', 70000,
                       'Number of examples from dataset extract to analysis the feature')
  flags.DEFINE_string('input_sentence', 'Halo.',
                      'The input sentence to inference,')

def flags_dict():
  """Define the flags.

  Returns:
    Command line arguments as Flags.
  """
  kwargs = {
      'file_path': FLAGS.file_path,
      'num_examples': FLAGS.num_examples,
      'embedding_dim': FLAGS.embedding_dim,
      'enc_units': FLAGS.enc_units,
      'dec_units': FLAGS.dec_units,
      'input_sentence': FLAGS.input_sentence
  }
  return kwargs


def run_main(argv):
  del argv
  kwargs = flags_dict()
  main(**kwargs)

def main(file_path, num_examples=70000, embedding_dim=256,
         enc_units=1024, dec_units=1024, input_sentence='Halo.'):
  input_tensor, target_tensor, inp_lang, targ_lang = utils.load_dataset(
      file_path, num_examples)
  max_length_targ, max_length_inp = \
      utils.max_length(target_tensor), utils.max_length(input_tensor)
  vocab_inp_size = len(inp_lang.word_index) + 1
  vocab_tar_size = len(targ_lang.word_index) + 1
  encoder = nmt.Encoder(vocab_inp_size, embedding_dim, enc_units)
  decoder = nmt.Decoder(vocab_tar_size, embedding_dim, dec_units)
  optimizer = tf.keras.optimizers.Adam()
  checkpoint_dir = os.path.join(sys.path[0] + '/training_checkpoints')
  checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                   encoder=encoder,
                                   decoder=decoder)
  _ = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
  eva_obj = Evaluate(encoder, decoder,
                     inp_lang, targ_lang, max_length_targ, max_length_inp)
  print("*********start evaluate**********")
  result, sentence = eva_obj.evaluate(input_sentence)
  print("{}:{}".format(sentence, result))

if __name__ == "__main__":
  evaluate_flags()
  app.run(run_main)

