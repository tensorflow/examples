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
"""Text classification demo code of Model Maker for TFLite."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app
from absl import flags
from absl import logging

import tensorflow as tf

from tflite_model_maker import model_spec
from tflite_model_maker import text_classifier
from tflite_model_maker.config import ExportFormat

FLAGS = flags.FLAGS


def define_flags():
  flags.DEFINE_string('export_dir', None,
                      'The directory to save exported files.')
  flags.DEFINE_string('spec', 'bert_classifier', 'The text classifier to run.')
  flags.mark_flag_as_required('export_dir')


def download_demo_data(**kwargs):
  """Downloads demo data, and returns directory path."""
  data_path = tf.keras.utils.get_file(
      fname='SST-2.zip',
      origin='https://dl.fbaipublicfiles.com/glue/data/SST-2.zip',
      extract=True,
      **kwargs)
  return os.path.join(os.path.dirname(data_path), 'SST-2')  # folder name


def run(data_dir, export_dir, spec='bert_classifier', **kwargs):
  """Runs demo."""
  # Chooses model specification that represents model.
  spec = model_spec.get(spec)

  # Gets training data and validation data.
  train_data = text_classifier.DataLoader.from_csv(
      filename=os.path.join(os.path.join(data_dir, 'train.tsv')),
      text_column='sentence',
      label_column='label',
      model_spec=spec,
      delimiter='\t',
      is_training=True)
  validation_data = text_classifier.DataLoader.from_csv(
      filename=os.path.join(os.path.join(data_dir, 'dev.tsv')),
      text_column='sentence',
      label_column='label',
      model_spec=spec,
      delimiter='\t',
      is_training=False)

  # Fine-tunes the model.
  model = text_classifier.create(
      train_data, model_spec=spec, validation_data=validation_data, **kwargs)

  # Gets evaluation results.
  _, acc = model.evaluate(validation_data)
  print('Eval accuracy: %f' % acc)

  # Exports to TFLite and SavedModel, with label and vocab files.
  export_format = [
      ExportFormat.TFLITE,
      ExportFormat.SAVED_MODEL,
  ]
  model.export(export_dir, export_format=export_format)


def main(_):
  logging.set_verbosity(logging.INFO)
  data_dir = download_demo_data()
  run(data_dir, FLAGS.export_dir, spec=FLAGS.spec)


if __name__ == '__main__':
  define_flags()
  app.run(main)
