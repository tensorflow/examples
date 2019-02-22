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
from tensorflow_examples.test_models.densenet import densenet
from tensorflow_examples.test_models.densenet import train


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

    train_obj = train.Train(epochs, enable_function)
    model = densenet.DenseNet(
        mode, growth_rate, output_classes, depth_of_model, num_of_blocks,
        data_format)
    train_obj.custom_loop(train_dataset, test_dataset, model)

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

    train_obj = train.Train(epochs, enable_function)
    model = densenet.DenseNet(
        mode, growth_rate, output_classes, depth_of_model, num_of_blocks,
        data_format)
    train_obj.keras_fit(train_dataset, test_dataset, model)


class DenseNetBenchmark(tf.test.Benchmark):

  def __init__(self, output_dir=None, **kwargs):
    self.output_dir = output_dir

  def benchmark_with_function_custom_loops(self):
    kwargs = {'epochs': 1, 'enable_function': True, 'buffer_size': 50000,
              'batch_size': 64, 'depth_of_model': 40, 'growth_rate': 12,
              'num_of_blocks': 3, 'output_classes': 10, 'mode': 'from_depth',
              'data_format': 'channels_last', 'dropout_rate': 0.2}
    self._run_and_report_benchmark(**kwargs)

  def benchmark_without_function_custom_loops(self):
    kwargs = {'epochs': 1, 'enable_function': False, 'buffer_size': 50000,
              'batch_size': 64, 'depth_of_model': 40, 'growth_rate': 12,
              'num_of_blocks': 3, 'output_classes': 10, 'mode': 'from_depth',
              'data_format': 'channels_last', 'dropout_rate': 0.2}
    self._run_and_report_benchmark(**kwargs)

  def benchmark_with_keras_fit(self):
    kwargs = {'epochs': 1, 'enable_function': True, 'buffer_size': 50000,
              'batch_size': 64, 'depth_of_model': 40, 'growth_rate': 12,
              'num_of_blocks': 3, 'output_classes': 10, 'mode': 'from_depth',
              'data_format': 'channels_last', 'dropout_rate': 0.2,
              'train_mode': 'keras_fit'}
    self._run_and_report_benchmark(**kwargs)

  def benchmark_with_function_custom_loops_300_epochs(self):
    kwargs = {'epochs': 300, 'enable_function': True, 'buffer_size': 50000,
              'batch_size': 64, 'depth_of_model': 40, 'growth_rate': 12,
              'num_of_blocks': 3, 'output_classes': 10, 'mode': 'from_depth',
              'data_format': 'channels_last', 'dropout_rate': 0.2,
              'bottleneck': False, 'compression': 1.}
    self._run_and_report_benchmark(**kwargs)

  def benchmark_with_keras_fit_300_epochs(self):
    kwargs = {'epochs': 300, 'enable_function': True, 'buffer_size': 50000,
              'batch_size': 64, 'depth_of_model': 40, 'growth_rate': 12,
              'num_of_blocks': 3, 'output_classes': 10, 'mode': 'from_depth',
              'data_format': 'channels_last', 'dropout_rate': 0.2,
              'train_mode': 'keras_fit', 'bottleneck': False, 'compression': 1.}
    self._run_and_report_benchmark(**kwargs)

  def _run_and_report_benchmark(self, **kwargs):
    start_time_sec = time.time()
    train_loss, train_acc, test_loss, test_acc = train.main(**kwargs)
    wall_time_sec = time.time() - start_time_sec

    extras = {'loss': train_loss,
              'training_accuracy_top_1': train_acc,
              'accuracy_top_1': test_acc}

    self.report_benchmark(
        wall_time=wall_time_sec, extras=extras)

if __name__ == '__main__':
  assert tf.__version__.startswith('2')
  tf.test.main()
