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
import tensorflow as tf # TF2
from tensorflow_examples.models.nmt_with_attention import nmt
from tensorflow_examples.models.nmt_with_attention import utils
from tensorflow_examples.models.nmt_with_attention.train import Train
assert tf.__version__.startswith('2')

FLAGS = flags.FLAGS

# if additional flags are needed, define it here.
flags.DEFINE_integer('num_gpu', 1, 'Number of GPUs to use')


class DistributedTrain(Train):
  """Distributed Train class.

  Args:
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

  def training_loop(self, train_dist_dataset, test_dist_dataset,
                    strategy):
    """Custom training and testing loop.

    Args:
      train_dist_dataset: Training dataset created using strategy.
      test_dist_dataset: Testing dataset created using strategy.
      strategy: Distribution strategy

    Returns:
      train_loss, test_loss
    """

    def distributed_train_epoch(ds):
      total_loss = 0.
      num_train_batches = 0
      for one_batch in ds:
        per_replica_loss = strategy.experimental_run_v2(
            self.train_step, args=(one_batch,))
        total_loss += strategy.reduce(
            tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None)
        num_train_batches += 1
      return total_loss, num_train_batches

    def distributed_test_epoch(ds):
      for one_batch in ds:
        strategy.experimental_run_v2(
            self.test_step, args=(one_batch,))
      return self.test_loss_metric.result()

    if self.enable_function:
      distributed_train_epoch = tf.function(distributed_train_epoch)
      distributed_test_epoch = tf.function(distributed_test_epoch)

    template = 'Epoch: {}, Train Loss: {}, Test Loss: {}'

    for epoch in range(self.epochs):
      train_total_loss, num_train_batches = distributed_train_epoch(
          train_dist_dataset)
      test_total_loss = distributed_test_epoch(
          test_dist_dataset)

      print (template.format(epoch,
                             train_total_loss / num_train_batches,
                             test_total_loss))

    return (train_total_loss / num_train_batches,
            test_total_loss)


def run_main(argv):
  del argv
  kwargs = utils.flags_dict()
  kwargs.update({'num_gpu': FLAGS.num_gpu})
  main(**kwargs)


def main(epochs, enable_function, buffer_size, batch_size, download_path,
         num_examples=70000, embedding_dim=256, enc_units=1024, dec_units=1024,
         num_gpu=1):

  devices = ['/device:GPU:{}'.format(i) for i in range(num_gpu)]
  strategy = tf.distribute.MirroredStrategy(devices)
  num_replicas = strategy.num_replicas_in_sync

  with strategy.scope():
    file_path = utils.download(download_path)
    train_ds, test_ds, inp_lang, targ_lang = utils.create_dataset(
        file_path, num_examples, buffer_size, batch_size)
    vocab_inp_size = len(inp_lang.word_index) + 1
    vocab_tar_size = len(targ_lang.word_index) + 1

    train_dist_dataset = strategy.experimental_distribute_dataset(train_ds)
    test_dist_dataset = strategy.experimental_distribute_dataset(test_ds)

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
    return train_obj.training_loop(train_dist_dataset,
                                   test_dist_dataset,
                                   strategy)

if __name__ == '__main__':
  utils.nmt_flags()
  app.run(run_main)
