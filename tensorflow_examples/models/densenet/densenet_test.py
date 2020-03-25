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
from tensorflow_examples.models.densenet import densenet
from tensorflow_examples.models.densenet import train
from tensorflow_examples.models.densenet import utils


def create_sample_dataset(batch_size):
  input_image = tf.random.uniform((32, 32, 3))
  label = tf.zeros((1,))
  dataset = tf.data.Dataset.from_tensors(
      (input_image, label)).batch(batch_size)
  return dataset


class DensenetTest(tf.test.TestCase):

  def test_one_epoch_with_function_custom_loop(self):
    epochs = 1
    enable_function = True
    depth_of_model = 7
    growth_rate = 2
    num_of_blocks = 3
    output_classes = 10
    mode = 'from_depth'
    data_format = 'channels_last'

    train_dataset = create_sample_dataset(batch_size=1)
    test_dataset = create_sample_dataset(batch_size=1)

    model = densenet.DenseNet(
        mode, growth_rate, output_classes, depth_of_model, num_of_blocks,
        data_format)
    train_obj = train.Train(epochs, enable_function, model)
    train_obj.custom_loop(train_dataset, test_dataset)

  def test_one_epoch_with_keras_fit(self):
    epochs = 1
    enable_function = True
    depth_of_model = 7
    growth_rate = 2
    num_of_blocks = 3
    output_classes = 10
    mode = 'from_depth'
    data_format = 'channels_last'

    train_dataset = create_sample_dataset(batch_size=1)
    test_dataset = create_sample_dataset(batch_size=1)

    model = densenet.DenseNet(
        mode, growth_rate, output_classes, depth_of_model, num_of_blocks,
        data_format)
    train_obj = train.Train(epochs, enable_function, model)
    train_obj.keras_fit(train_dataset, test_dataset)


class DenseNetBenchmark(tf.test.Benchmark):

  def __init__(self, output_dir=None, **kwargs):
    self.output_dir = output_dir

  def benchmark_with_function_custom_loops(self):
    kwargs = utils.get_cifar10_kwargs()
    self._run_and_report_benchmark(**kwargs)

  def benchmark_without_function_custom_loops(self):
    kwargs = utils.get_cifar10_kwargs()
    kwargs.update({'enable_function': False})
    self._run_and_report_benchmark(**kwargs)

  def benchmark_with_keras_fit(self):
    kwargs = utils.get_cifar10_kwargs()
    kwargs.update({'train_mode': 'keras_fit'})
    self._run_and_report_benchmark(**kwargs)

  def benchmark_with_function_custom_loops_300_epochs(self):
    kwargs = utils.get_cifar10_kwargs()
    kwargs.update({'epochs': 300, 'data_format': 'channels_first',
                   'bottleneck': False, 'compression': 1.})
    self._run_and_report_benchmark(**kwargs)

  def benchmark_with_keras_fit_300_epochs(self):
    kwargs = utils.get_cifar10_kwargs()
    kwargs.update({'epochs': 300, 'data_format': 'channels_first',
                   'train_mode': 'keras_fit', 'bottleneck': False,
                   'compression': 1.})
    self._run_and_report_benchmark(**kwargs)

  def _run_and_report_benchmark(self, **kwargs):
    """Run the benchmark and report metrics.report.

    Args:
      **kwargs: All args passed to the test.
    """
    start_time_sec = time.time()
    train_loss, train_acc, _, test_acc = train.main(**kwargs)
    wall_time_sec = time.time() - start_time_sec

    metrics = []
    metrics.append({'name': 'accuracy_top_1',
                    'value': test_acc,
                    'min_value': .944,
                    'max_value': .949})

    metrics.append({'name': 'training_accuracy_top_1',
                    'value': train_acc})

    metrics.append({'name': 'train_loss',
                    'value': train_loss})

    self.report_benchmark(wall_time=wall_time_sec, metrics=metrics)

if __name__ == '__main__':
  tf.test.main()
