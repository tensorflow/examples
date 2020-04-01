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
"""Distributed Train.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

import tensorflow as tf
from tensorflow_examples.models.nmt_with_attention import nmt
from tensorflow_examples.models.nmt_with_attention import utils
from tensorflow_examples.models.nmt_with_attention.train import Train


FLAGS = flags.FLAGS


class DistributedTrain(Train):
  """Distributed Train class.

  Attributes:
    epochs: Number of epochs.
    enable_function: Decorate function with tf.function.
    encoder: Encoder.
    decoder: Decoder.
    inp_lang: Input language tokenizer.
    targ_lang: Target language tokenizer.
    batch_size: Batch size.
    per_replica_batch_size: Batch size per replica for sync replicas.
  """

  def __init__(self, epochs, enable_function, encoder, decoder, inp_lang,
               targ_lang, batch_size, per_replica_batch_size):
    Train.__init__(
        self, epochs, enable_function, encoder, decoder, inp_lang, targ_lang,
        batch_size, per_replica_batch_size)

  def training_loop(self, train_iterator, test_iterator,
                    num_train_steps_per_epoch, num_test_steps_per_epoch,
                    strategy):
    """Custom training and testing loop.

    Args:
      train_iterator: Training iterator created using strategy
      test_iterator: Testing iterator created using strategy
      num_train_steps_per_epoch: number of training steps in an epoch.
      num_test_steps_per_epoch: number of test steps in an epoch.
      strategy: Distribution strategy

    Returns:
      train_loss, test_loss
    """

    # this code is expected to change.
    def distributed_train():
      return strategy.experimental_run(self.train_step, train_iterator)

    def distributed_test():
      return strategy.experimental_run(self.test_step, test_iterator)

    if self.enable_function:
      distributed_train = tf.function(distributed_train)
      distributed_test = tf.function(distributed_test)

    template = 'Epoch: {}, Train Loss: {}, Test Loss: {}'

    for epoch in range(self.epochs):
      self.train_loss_metric.reset_states()
      self.test_loss_metric.reset_states()

      train_iterator.initialize()
      for _ in range(num_train_steps_per_epoch):
        distributed_train()

      test_iterator.initialize()
      for _ in range(num_test_steps_per_epoch):
        distributed_test()

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

  strategy = tf.distribute.MirroredStrategy()
  num_replicas = strategy.num_replicas_in_sync

  file_path = utils.download(download_path)
  train_ds, test_ds, inp_lang, targ_lang = utils.create_dataset(
      file_path, num_examples, buffer_size, batch_size)

  with strategy.scope():
    vocab_inp_size = len(inp_lang.word_index) + 1
    vocab_tar_size = len(targ_lang.word_index) + 1

    num_train_steps_per_epoch = tf.data.experimental.cardinality(train_ds)
    num_test_steps_per_epoch = tf.data.experimental.cardinality(test_ds)

    train_iterator = strategy.make_dataset_iterator(train_ds)
    test_iterator = strategy.make_dataset_iterator(test_ds)

    local_batch_size, remainder = divmod(batch_size, num_replicas)

    template = ('Batch size ({}) must be divisible by the '
                'number of replicas ({})')
    if remainder:
      raise ValueError(template.format(batch_size, num_replicas))

    encoder = nmt.Encoder(vocab_inp_size, embedding_dim, enc_units,
                          local_batch_size)
    decoder = nmt.Decoder(vocab_tar_size, embedding_dim, dec_units)

    train_obj = DistributedTrain(epochs, enable_function, encoder, decoder,
                                 inp_lang, targ_lang, batch_size,
                                 local_batch_size)
    print ('Training ...')
    return train_obj.training_loop(train_iterator,
                                   test_iterator,
                                   num_train_steps_per_epoch,
                                   num_test_steps_per_epoch,
                                   strategy)

if __name__ == '__main__':
  utils.nmt_flags()
  app.run(run_main)
