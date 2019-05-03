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
import tensorflow as tf # TF2
from tensorflow_examples.models.nmt_with_attention import distributed_train
from tensorflow_examples.models.nmt_with_attention import utils
assert tf.__version__.startswith('2')


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
    kwargs.update({'epochs': 10, 'num_gpu': 2, 'batch_size': 128})
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
