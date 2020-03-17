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
"""Tests for distributed nmt_with_attention."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
from tensorflow_examples.models.nmt_with_attention import distributed_train
from tensorflow_examples.models.nmt_with_attention import utils


class NmtDistributedTest(tf.test.TestCase):

  def test_one_epoch_multi_device(self):
    if tf.test.is_gpu_available():
      print('Using 2 virtual GPUs.')
      device = tf.config.experimental.list_physical_devices('GPU')[0]
      tf.config.experimental.set_virtual_device_configuration(
          device, [
              tf.config.experimental.VirtualDeviceConfiguration(
                  memory_limit=8192),
              tf.config.experimental.VirtualDeviceConfiguration(
                  memory_limit=8192)
          ])

    kwargs = utils.get_common_kwargs()
    kwargs.update({
        'epochs': 1,
        'batch_size': 16,
        'num_examples': 10,
        'embedding_dim': 4,
        'enc_units': 4,
        'dec_units': 4
    })

    distributed_train.main(**kwargs)


class NmtDistributedBenchmark(tf.test.Benchmark):

  def __init__(self, output_dir=None, **kwargs):
    self.output_dir = output_dir

  def benchmark_one_epoch_1_gpu(self):
    kwargs = utils.get_common_kwargs()
    kwargs.update({'enable_function': False})
    self._run_and_report_benchmark(**kwargs)

  def benchmark_one_epoch_1_gpu_function(self):
    kwargs = utils.get_common_kwargs()
    self._run_and_report_benchmark(**kwargs)

  def benchmark_ten_epochs_2_gpus(self):
    kwargs = utils.get_common_kwargs()
    kwargs.update({'epochs': 10, 'batch_size': 128})
    self._run_and_report_benchmark(**kwargs)

  def _run_and_report_benchmark(self, **kwargs):
    start_time_sec = time.time()
    train_loss, test_loss = distributed_train.main(**kwargs)
    wall_time_sec = time.time() - start_time_sec

    extras = {'train_loss': train_loss,
              'test_loss': test_loss}

    self.report_benchmark(
        wall_time=wall_time_sec, extras=extras)

if __name__ == '__main__':
  tf.test.main()
