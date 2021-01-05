# Copyright 2020 Google Research. All Rights Reserved.
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
r"""Simple tools for TensorRT.

Example usage:

$ export ROOT=/tmp/d4
$ python model_inspect.py --runmode=freeze --model_name=efficientdet-d4 \
    --logdir=$ROOT  # --hparams=xyz.yaml
$ python tensorrt.py --tf_savedmodel_dir=$ROOT/savedmodel \
    --trt_savedmodel_dir=$ROOT/trtmodel
"""
import time
from absl import app
from absl import flags
import numpy as np
import tensorflow.compat.v1 as tf
# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.compiler.tensorrt import trt_convert as trt

flags.DEFINE_string('tf_savedmodel_dir', None, 'TensorFlow saved model dir.')
flags.DEFINE_string('trt_savedmodel_dir', None, 'TensorRT saved model dir.')
FLAGS = flags.FLAGS


def convert2trt(tf_savedmodel_dir: str, trt_savedmodel_dir: str):
  converter = trt.TrtGraphConverter(
      input_saved_model_dir=tf_savedmodel_dir,
      max_workspace_size_bytes=(2 << 20),
      precision_mode='FP16',
      maximum_cached_engines=1)
  converter.convert()
  converter.save(trt_savedmodel_dir)


def benchmark(trt_savedmodel_dir: str, warmup_runs: int = 5, bm_runs: int = 20):
  """Benchmark TRT latency for a given TRT saved model."""
  with tf.Session() as sess:
    # First load the Saved Model into the session
    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING],
                               trt_savedmodel_dir)
    graph = tf.get_default_graph()
    input_shape = graph.get_tensor_by_name('input:0').shape
    x = np.ones(input_shape).astype(np.float32)
    ss = lambda i: '' if i == 0 else '_%d' % i
    outputs = ['box_net/box-predict%s/BiasAdd:0' % ss(i) for i in range(1)]
    outputs += ['class_net/class-predict%s/BiasAdd:0' % ss(i) for i in range(5)]
    # Apply reduce_sum to avoid massive data move between GPU and CPU.
    outputs = [tf.reduce_sum(graph.get_tensor_by_name(i)) for i in outputs]

    # warmup
    for _ in range(warmup_runs):
      sess.run(outputs, feed_dict={'input:0': x})
    # benchmark
    s = time.perf_counter()
    for _ in range(bm_runs):
      sess.run(outputs, feed_dict={'input:0': x})
    e = time.perf_counter()
    print('Benchmark latency=%.4f  FPS=%.2f', (e - s) / bm_runs,
          bm_runs / (e - s))


def main(_):
  if FLAGS.tf_savedmodel_dir:
    convert2trt(FLAGS.tf_savedmodel_dir, FLAGS.trt_savedmodel_dir)
  benchmark(FLAGS.trt_savedmodel_dir, FLAGS.warmup_runs, FLAGS.bm_runs)


if __name__ == '__main__':
  flags.mark_flag_as_required('trt_savedmodel_dir')
  tf.disable_v2_behavior()
  app.run(main)
