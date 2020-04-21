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
"""Tests for nmt_with_attention."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import tensorflow as tf
from tensorflow_examples.models.nmt_with_attention import train
from tensorflow_examples.models.nmt_with_attention import utils


class NmtTest(tf.test.TestCase):

  def test_one_epoch(self):
    num_examples = 10
    buffer_size = 10
    batch_size = 1
    embedding_dim = 4
    enc_units = 4
    dec_units = 4
    epochs = 1

    train.main(epochs, True, buffer_size, batch_size, 'datasets', num_examples,
               embedding_dim, enc_units, dec_units)


class NmtBenchmark(tf.test.Benchmark):

  def __init__(self, output_dir=None, **kwargs):
    self.output_dir = output_dir

  def benchmark_one_epoch(self):
    kwargs = utils.get_common_kwargs()
    self._run_and_report_benchmark(**kwargs)

  def benchmark_ten_epochs(self):
    kwargs = utils.get_common_kwargs()
    kwargs.update({'epochs': 10})
    self._run_and_report_benchmark(**kwargs)

  def _run_and_report_benchmark(self, **kwargs):
    start_time_sec = time.time()
    train_loss, test_loss = train.main(**kwargs)
    wall_time_sec = time.time() - start_time_sec

    extras = {'train_loss': train_loss,
              'test_loss': test_loss}

    self.report_benchmark(
        wall_time=wall_time_sec, extras=extras)

if __name__ == '__main__':
  tf.test.main()
