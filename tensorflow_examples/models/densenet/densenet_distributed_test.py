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
import tensorflow as tf
from tensorflow_examples.models.densenet import distributed_train
from tensorflow_examples.models.densenet import utils


class DenseNetDistributedBenchmark(tf.test.Benchmark):

  def __init__(self, output_dir=None, **kwargs):
    self.output_dir = output_dir

  def benchmark_with_function_custom_loops(self):
    kwargs = utils.get_cifar10_kwargs()
    self._run_and_report_benchmark(**kwargs)

  def benchmark_with_function_custom_loops_300_epochs_2_gpus(self):
    kwargs = utils.get_cifar10_kwargs()
    kwargs.update({'epochs': 300, 'data_format': 'channels_first',
                   'bottleneck': False, 'compression': 1., 'num_gpu': 2,
                   'batch_size': 128})

    self._run_and_report_benchmark(**kwargs)

  def benchmark_with_function_custom_loops_300_epochs_8_gpus(self):
    kwargs = utils.get_cifar10_kwargs()
    kwargs.update({'epochs': 300, 'data_format': 'channels_first',
                   'bottleneck': False, 'compression': 1., 'num_gpu': 8,
                   'batch_size': 512})

    self._run_and_report_benchmark(**kwargs)

  def _run_and_report_benchmark(self, top_1_min=.944, top_1_max=.949, **kwargs):
    """Run the benchmark and report metrics.report.

    Args:
      top_1_min: Min value for top_1 accuracy.  Default range is SOTA.
      top_1_max: Max value for top_1 accuracy.
      **kwargs: All args passed to the test.
    """
    start_time_sec = time.time()
    train_loss, train_acc, _, test_acc = distributed_train.main(**kwargs)
    wall_time_sec = time.time() - start_time_sec

    metrics = []
    metrics.append({'name': 'accuracy_top_1',
                    'value': test_acc,
                    'min_value': top_1_min,
                    'max_value': top_1_max})

    metrics.append({'name': 'training_accuracy_top_1',
                    'value': train_acc})

    metrics.append({'name': 'train_loss',
                    'value': train_loss})

    self.report_benchmark(wall_time=wall_time_sec, metrics=metrics)

if __name__ == '__main__':
  tf.test.main()
