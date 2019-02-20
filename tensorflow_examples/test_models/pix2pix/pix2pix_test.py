"""Tests for Pix2Pix."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
import tensorflow as tf # TF2
from tensorflow_examples.test_models.pix2pix import data_download
from tensorflow_examples.test_models.pix2pix import pix2pix

FLAGS = flags.FLAGS


class Pix2PixTest(tf.test.TestCase):

  def test_one_step_with_function(self):
    epochs = 1
    batch_size = 1
    enable_function = True

    input_image = tf.random.uniform((256, 256, 3))
    target_image = tf.random.uniform((256, 256, 3))

    train_dataset = tf.data.Dataset.from_tensors(
        (input_image, target_image)).batch(batch_size)
    checkpoint_pr = pix2pix.get_checkpoint_prefix()

    pix2pix_obj = pix2pix.Pix2pix(epochs, enable_function)
    pix2pix_obj.train(train_dataset, checkpoint_pr)

  def test_one_step_without_function(self):
    epochs = 1
    batch_size = 1
    enable_function = False

    input_image = tf.random.uniform((256, 256, 3))
    target_image = tf.random.uniform((256, 256, 3))

    train_dataset = tf.data.Dataset.from_tensors(
        (input_image, target_image)).batch(batch_size)

    pix2pix_obj = pix2pix.Pix2pix(epochs, enable_function)

    checkpoint_pr = pix2pix.get_checkpoint_prefix()
    pix2pix_obj.train(train_dataset, checkpoint_pr)


class Pix2PixBenchmark(tf.test.Benchmark):

  def __init__(self, output_dir=None):
    self.output_dir = output_dir

  def benchmark_with_function(self):
    path = data_download.main("datasets")
    kwargs = {"epochs": 6, "enable_function": True, "path": path,
              "buffer_size": 400, "batch_size": 1}
    self._run_and_report_benchmark(**kwargs)

  def benchmark_without_function(self):
    path = data_download.main("datasets")
    kwargs = {"epochs": 6, "enable_function": False, "path": path,
              "buffer_size": 400, "batch_size": 1}
    self._run_and_report_benchmark(**kwargs)

  def _run_and_report_benchmark(self, **kwargs):
    time_list = pix2pix.main(**kwargs)
    # 1st epoch is the warmup epoch hence skipping it for calculating time.
    self.report_benchmark(wall_time=tf.reduce_mean(time_list[1:]))

if __name__ == "__main__":
  assert tf.__version__.startswith('2')
  tf.test.main()
