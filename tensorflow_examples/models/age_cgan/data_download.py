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
"""Download the data.

Dataset Citation:
@article{Rothe-IJCV-2016,
  author = {Rasmus Rothe and Radu Timofte and Luc Van Gool},
  title = {Deep expectation of real and apparent age from a single image without facial landmarks},
  journal = {International Journal of Computer Vision (IJCV)},
  year = {2016},
  month = {July},
}
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import tensorflow as tf
assert tf.__version__.startswith('2')

ap = argparse.ArgumentParser()
ap.add_argument('-dp', '--download_path', required=False, help='Download Path')
args = vars(ap.parse_args())

data_url = "https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/wiki_crop.tar"

def download_data(download_path):
	path_to_zip = tf.keras.utils.get_file(
		'wiki_crop.tar', cache_subdir=download_path,
		origin = data_url, extract=True)

	path_to_folder = os.path.join(os.path.dirname(path_to_zip), '')

	return path_to_folder

if __name__ == '__main__':	
	if args['download_path'] is not None:
		path = download_data(args["download_path"])
	else:
		cur = os.getcwd()
		path = download_data(cur)
