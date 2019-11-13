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
"""Text classification demo code of model customization for TFLite."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app
from absl import flags
from absl import logging

import tensorflow as tf # TF2
from tensorflow_examples.lite.model_customization.core.data_util.text_dataloader import TextClassifierDataLoader
from tensorflow_examples.lite.model_customization.core.model_export_format import ModelExportFormat
from tensorflow_examples.lite.model_customization.core.task import text_classifier

flags.DEFINE_string('tflite_filename', None, 'File name to save tflite model.')
flags.DEFINE_string('label_filename', None, 'File name to save labels.')
flags.DEFINE_string('vocab_filename', None, 'File name to save vocabulary.')
FLAGS = flags.FLAGS


def main(_):
  logging.set_verbosity(logging.INFO)

  data_path = tf.keras.utils.get_file(
      fname='aclImdb',
      origin='http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz',
      untar=True)
  train_data = TextClassifierDataLoader.from_folder(
      filename=os.path.join(os.path.join(data_path, 'train')),
      class_labels=['pos', 'neg'])
  test_data = TextClassifierDataLoader.from_folder(
      filename=os.path.join(data_path, 'test'))

  model = text_classifier.create(
      train_data, model_export_format=ModelExportFormat.TFLITE)

  _, acc = model.evaluate(test_data)
  print('Test accuracy: %f' % acc)

  model.export(FLAGS.tflite_filename, FLAGS.label_filename,
               FLAGS.vocab_filename)


if __name__ == '__main__':
  tf.compat.v1.enable_v2_behavior()
  flags.mark_flag_as_required('tflite_filename')
  flags.mark_flag_as_required('label_filename')
  flags.mark_flag_as_required('vocab_filename')
  app.run(main)
