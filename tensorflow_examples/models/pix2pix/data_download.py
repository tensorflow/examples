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
"""Download facades data.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl import app
from absl import flags
import tensorflow as tf # TF2
assert tf.__version__.startswith('2')

FLAGS = flags.FLAGS

flags.DEFINE_string('download_path', 'datasets', 'Download folder')

_URL = 'https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/facades.tar.gz'


def _main(argv):
  del argv
  download_path = FLAGS.download_path
  main(download_path)


def main(download_path):
  path_to_zip = tf.keras.utils.get_file(
      'facades.tar.gz', cache_subdir=download_path,
      origin=_URL, extract=True)

  path_to_folder = os.path.join(os.path.dirname(path_to_zip), 'facades/')

  return path_to_folder

if __name__ == '__main__':
  app.run(_main)
