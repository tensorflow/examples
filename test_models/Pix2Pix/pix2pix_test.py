"""Tests for Pix2Pix."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from absl import flags
import tensorflow as tf # TF2
from tensorflow_examples.test_models.Pix2Pix import data_download
from tensorflow_examples.test_models.Pix2Pix import pix2pix

FLAGS = flags.FLAGS


class Pix2pixTest(tf.test.TestCase):

  def test_one_step(self):
    epochs = 1
    input_image = tf.random.uniform((256, 256, 3))
    target_image = tf.random.uniform((256, 256, 3))

    train_dataset = tf.data.Dataset.from_tensors(
        (input_image, target_image)).batch(1)

    gen = pix2pix.generator_model()
    disc = pix2pix.discriminator_model()
    checkpoint, checkpoint_pr = pix2pix.get_checkpoint(gen, disc)
    pix2pix.train(train_dataset, gen, disc, checkpoint, checkpoint_pr, epochs)


class Pix2PixBenchmark(tf.test.Benchmark):

  def __init__(self, output_dir=None):
    self.output_dir = output_dir

  def benchmark_with_function(self):
    path = data_download.main("datasets")
    kwargs = {"epochs": 1, "enable_function": True, "path": path,
              "buffer_size": 400, "batch_size": 1}
    self._run_and_report_benchmark(**kwargs)

  def benchmark_without_function(self):
    path = data_download.main("datasets")
    kwargs = {"epochs": 1, "enable_function": False, "path": path,
              "buffer_size": 400, "batch_size": 1}
    self._run_and_report_benchmark(**kwargs)

  def _run_and_report_benchmark(self, **kwargs):
    start_time_sec = time.time()
    pix2pix.main(**kwargs)
    wall_time_sec = time.time() - start_time_sec

    self.report_benchmark(wall_time=wall_time_sec)

if __name__ == "__main__":
  assert tf.__version__.startswith('2')
  tf.test.main()
