# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Densely Connected Convolutional Networks.

Reference [
Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf # TF2
from tensorflow_examples.models.densenet import distributed_train


def get_cifar10_kwargs():
  return {'epochs': 1, 'enable_function': True, 'buffer_size': 50000,
          'batch_size': 64, 'depth_of_model': 40, 'growth_rate': 12,
          'num_of_blocks': 3, 'output_classes': 10, 'mode': 'from_depth',
          'data_format': 'channels_last', 'dropout_rate': 0.}


class DenseNetDistributedBenchmark(tf.test.Benchmark):

  def __init__(self, output_dir=None, **kwargs):
    self.output_dir = output_dir

  def benchmark_with_function_custom_loops(self):
    kwargs = get_cifar10_kwargs()
    self._run_and_report_benchmark(**kwargs)

  def benchmark_with_function_custom_loops_300_epochs_2_gpus(self):
    kwargs = get_cifar10_kwargs()
    kwargs.update({'epochs': 300, 'data_format': 'channels_first',
                   'bottleneck': False, 'compression': 1., 'num_gpu': 2})

    self._run_and_report_benchmark(**kwargs)

  def benchmark_with_function_custom_loops_300_epochs_8_gpus(self):
    kwargs = get_cifar10_kwargs()
    kwargs.update({'epochs': 300, 'data_format': 'channels_first',
                   'bottleneck': False, 'compression': 1., 'num_gpu': 8})

    self._run_and_report_benchmark(**kwargs)

  def _run_and_report_benchmark(self, **kwargs):
    start_time_sec = time.time()
    train_loss, train_acc, _, test_acc = distributed_train.main(**kwargs)
    wall_time_sec = time.time() - start_time_sec

    extras = {'train_loss': train_loss,
              'training_accuracy_top_1': train_acc,
              'accuracy_top_1': test_acc}

    self.report_benchmark(
        wall_time=wall_time_sec, extras=extras)

if __name__ == '__main__':
  assert tf.__version__.startswith('2')
  tf.test.main()
