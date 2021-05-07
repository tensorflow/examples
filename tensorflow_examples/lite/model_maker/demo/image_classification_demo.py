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
"""Image classification demo code of Model Maker for TFLite."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app
from absl import flags
from absl import logging

import tensorflow as tf

from tflite_model_maker import image_classifier
from tflite_model_maker import model_spec
from tflite_model_maker.config import ExportFormat

FLAGS = flags.FLAGS


def define_flags():
  flags.DEFINE_string('export_dir', None,
                      'The directory to save exported files.')
  flags.DEFINE_string('spec', 'efficientnet_lite0',
                      'The image classifier to run.')
  flags.mark_flag_as_required('export_dir')


def download_demo_data(**kwargs):
  """Downloads demo data, and returns directory path."""
  data_dir = tf.keras.utils.get_file(
      fname='flower_photos.tgz',
      origin='https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
      extract=True,
      **kwargs)
  return os.path.join(os.path.dirname(data_dir), 'flower_photos')  # folder name


def run(data_dir, export_dir, spec='efficientnet_lite0', **kwargs):
  """Runs demo."""
  spec = model_spec.get(spec)
  data = image_classifier.DataLoader.from_folder(data_dir)
  train_data, rest_data = data.split(0.8)
  validation_data, test_data = rest_data.split(0.5)

  model = image_classifier.create(
      train_data, model_spec=spec, validation_data=validation_data, **kwargs)

  _, acc = model.evaluate(test_data)
  print('Test accuracy: %f' % acc)

  # Exports to TFLite and SavedModel, with label file.
  export_format = [
      ExportFormat.TFLITE,
      ExportFormat.SAVED_MODEL,
  ]
  model.export(export_dir, export_format=export_format)


def main(_):
  logging.set_verbosity(logging.INFO)
  data_dir = download_demo_data()
  export_dir = os.path.expanduser(FLAGS.export_dir)
  run(data_dir, export_dir, spec=FLAGS.spec)


if __name__ == '__main__':
  define_flags()
  app.run(main)
