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
