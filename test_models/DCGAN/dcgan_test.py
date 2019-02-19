"""DCGAN tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from absl import flags
import tensorflow as tf # TF2
from tensorflow_examples.test_models.DCGAN import dcgan

FLAGS = flags.FLAGS


class DcganTest(tf.test.TestCase):

  def test_one_epoch_with_function(self):
    epochs = 1
    batch_size = 1
    enable_function = True

    input_image = tf.random.uniform((28, 28, 1))
    label = tf.zeros((1,))
    train_dataset = tf.data.Dataset.from_tensors(
        (input_image, label)).batch(batch_size)
    checkpoint_pr = dcgan.get_checkpoint_prefix()

    dcgan_obj = dcgan.Dcgan(epochs, enable_function, batch_size)
    dcgan_obj.train(train_dataset, checkpoint_pr)

  def test_one_epoch_without_function(self):
    epochs = 1
    batch_size = 1
    enable_function = False

    input_image = tf.random.uniform((28, 28, 1))
    label = tf.zeros((1,))
    train_dataset = tf.data.Dataset.from_tensors(
        (input_image, label)).batch(batch_size)
    checkpoint_pr = dcgan.get_checkpoint_prefix()

    dcgan_obj = dcgan.Dcgan(epochs, enable_function, batch_size)
    dcgan_obj.train(train_dataset, checkpoint_pr)


class DCGANBenchmark(tf.test.Benchmark):

  def __init__(self, output_dir=None):
    self.output_dir = output_dir

  def benchmark_with_function(self):
    kwargs = {"epochs": 1, "enable_function": True,
              "buffer_size": 10000, "batch_size": 64}
    self._run_and_report_benchmark(**kwargs)

  def benchmark_without_function(self):
    kwargs = {"epochs": 1, "enable_function": False,
              "buffer_size": 10000, "batch_size": 64}
    self._run_and_report_benchmark(**kwargs)

  def _run_and_report_benchmark(self, **kwargs):
    start_time_sec = time.time()
    dcgan.main(**kwargs)
    wall_time_sec = time.time() - start_time_sec

    self.report_benchmark(wall_time=wall_time_sec)

if __name__ == "__main__":
  assert tf.__version__.startswith('2')
  tf.test.main()
