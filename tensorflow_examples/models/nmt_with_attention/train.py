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
"""Train.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
import tensorflow as tf # TF2
from tensorflow_examples.models.nmt_with_attention import nmt
from tensorflow_examples.models.nmt_with_attention import utils
assert tf.__version__.startswith('2')


class Train(object):
  """Train class.

  Args:
    epochs: Number of epochs.
    enable_function: Decorate function with tf.function.
    encoder: Encoder.
    decoder: Decoder.
    inp_lang: Input language tokenizer.
    targ_lang: Target language tokenizer.
    batch_size: Batch size.
  """

  def __init__(self, epochs, enable_function, encoder, decoder, inp_lang,
               targ_lang, batch_size):
    self.epochs = epochs
    self.enable_function = enable_function
    self.encoder = encoder
    self.decoder = decoder
    self.inp_lang = inp_lang
    self.targ_lang = targ_lang
    self.batch_size = batch_size
    self.optimizer = tf.keras.optimizers.Adam()
    self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True)
    self.train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
    self.test_loss_metric = tf.keras.metrics.Mean(name='test_loss')

  def loss_function(self, real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = self.loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)

  def train_step(self, inputs):
    """One train step.

    Args:
      inputs: tuple of input tensor, target tensor.
    """

    loss = 0
    enc_hidden = self.encoder.initialize_hidden_state()

    inp, targ = inputs

    with tf.GradientTape() as tape:
      enc_output, enc_hidden = self.encoder(inp, enc_hidden)
      dec_hidden = enc_hidden
      dec_input = tf.expand_dims(
          [self.targ_lang.word_index['<start>']] * self.batch_size, 1)

      for t in range(1, targ.shape[1]):
        # passing enc_output to the decoder
        predictions, dec_hidden, _ = self.decoder(
            dec_input, dec_hidden, enc_output)
        loss += self.loss_function(targ[:, t], predictions)
        # using teacher forcing
        dec_input = tf.expand_dims(targ[:, t], 1)

    batch_loss = (loss / int(targ.shape[1]))
    variables = self.encoder.trainable_variables + self.decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    self.optimizer.apply_gradients(zip(gradients, variables))

    self.train_loss_metric(batch_loss)

  def test_step(self, inputs_test):
    """One test step.

    Args:
      inputs_test: tuple of input tensor, target tensor.
    """

    loss = 0
    enc_hidden = self.encoder.initialize_hidden_state()

    inp_test, targ_test = inputs_test

    enc_output, enc_hidden = self.encoder(inp_test, enc_hidden)
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims(
        [self.targ_lang.word_index['<start>']] * self.batch_size, 1)

    for t in range(1, targ_test.shape[1]):
      predictions, dec_hidden, _ = self.decoder(
          dec_input, dec_hidden, enc_output)
      loss += self.loss_function(targ_test[:, t], predictions)

      prediction_id = tf.argmax(predictions, axis=1)
      # passing the predictions back to the model as the input.
      dec_input = tf.expand_dims(prediction_id, 1)

    batch_loss = (loss / int(targ_test.shape[1]))

    self.test_loss_metric(batch_loss)

  def training_loop(self, train_ds, test_ds):
    """Custom training and testing loop.

    Args:
      train_ds: Training dataset
      test_ds: Testing dataset

    Returns:
      train_loss, test_loss
    """

    if self.enable_function:
      self.train_step = tf.function(self.train_step)
      self.test_step = tf.function(self.test_step)

    template = 'Epoch: {}, Train Loss: {}, Test Loss: {}'

    for epoch in range(self.epochs):
      self.train_loss_metric.reset_states()
      self.test_loss_metric.reset_states()

      for inp, targ in train_ds:
        self.train_step((inp, targ))

      for inp_test, targ_test in test_ds:
        self.test_step((inp_test, targ_test))

      print (template.format(epoch,
                             self.train_loss_metric.result().numpy(),
                             self.test_loss_metric.result().numpy()))

    return (self.train_loss_metric.result().numpy(),
            self.test_loss_metric.result().numpy())


def run_main(argv):
  del argv
  kwargs = utils.flags_dict()
  main(**kwargs)


def main(epochs, enable_function, buffer_size, batch_size, download_path,
         num_examples=70000, embedding_dim=256, enc_units=1024, dec_units=1024):
  file_path = utils.download(download_path)
  train_ds, test_ds, inp_lang, targ_lang = utils.create_dataset(
      file_path, num_examples, buffer_size, batch_size)
  vocab_inp_size = len(inp_lang.word_index) + 1
  vocab_tar_size = len(targ_lang.word_index) + 1

  encoder = nmt.Encoder(vocab_inp_size, embedding_dim, enc_units, batch_size)
  decoder = nmt.Decoder(vocab_tar_size, embedding_dim, dec_units, batch_size)

  train_obj = Train(epochs, enable_function, encoder, decoder,
                    inp_lang, targ_lang, batch_size)
  print ('Training ...')
  return train_obj.training_loop(train_ds, test_ds)

if __name__ == '__main__':
  utils.nmt_flags()
  app.run(run_main)
