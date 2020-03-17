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
from tensorflow_examples.lite.model_maker.core.data_util.image_dataloader import ImageClassifierDataLoader
from tensorflow_examples.lite.model_maker.core.model_export_format import ModelExportFormat
from tensorflow_examples.lite.model_maker.core.task import image_classifier
from tensorflow_examples.lite.model_maker.core.task import model_spec

FLAGS = flags.FLAGS


def define_flags():
  flags.DEFINE_string('tflite_filename', None,
                      'File name to save tflite model.')
  flags.DEFINE_string('label_filename', None, 'File name to save labels.')
  flags.mark_flag_as_required('tflite_filename')
  flags.mark_flag_as_required('label_filename')


def download_demo_data(**kwargs):
  """Downloads demo data, and returns directory path."""
  data_dir = tf.keras.utils.get_file(
      fname='flower_photos.tgz',
      origin='https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
      untar=True,
      **kwargs)
  return os.path.join(data_dir, '..', 'flower_photos')  # folder name


def run(data_dir,
        tflite_filename,
        label_filename,
        spec='efficientnet_lite0',
        **kwargs):
  """Runs demo."""
  spec = model_spec.get(spec)
  data = ImageClassifierDataLoader.from_folder(data_dir)
  train_data, rest_data = data.split(0.8)
  validation_data, test_data = rest_data.split(0.5)

  model = image_classifier.create(
      train_data,
      model_export_format=ModelExportFormat.TFLITE,
      model_spec=spec,
      validation_data=validation_data,
      **kwargs)

  _, acc = model.evaluate(test_data)
  print('Test accuracy: %f' % acc)
  model.export(tflite_filename, label_filename)


def main(_):
  logging.set_verbosity(logging.INFO)
  data_dir = download_demo_data()
  run(data_dir, FLAGS.tflite_filename, FLAGS.label_filename, epochs=10)


if __name__ == '__main__':
  define_flags()
  app.run(main)
