# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

from absl import app
from absl import flags
from absl import logging

import tensorflow.compat.v2 as tf
from tflite_model_maker import model_spec
from tflite_model_maker import question_answer

FLAGS = flags.FLAGS


def define_flags():
  flags.DEFINE_string('export_dir', None,
                      'The directory to save exported files.')
  flags.mark_flag_as_required('export_dir')
  flags.DEFINE_string('spec', 'bert_qa', 'The QA model to run.')


def download_demo_data(**kwargs):
  """Downloads demo data, and returns directory path."""
  train_data_path = tf.keras.utils.get_file(
      fname='train-v1.1.json',
      origin='https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json',
      **kwargs)
  validation_data_path = tf.keras.utils.get_file(
      fname='dev-v1.1.json',
      origin='https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json',
      **kwargs)
  return train_data_path, validation_data_path


def run(train_data_path,
        validation_data_path,
        export_dir,
        spec='bert_qa',
        **kwargs):
  """Runs demo."""
  # Chooses model specification that represents model.
  spec = model_spec.get(spec)

  # Gets training data and validation data.
  train_data = question_answer.DataLoader.from_squad(
      train_data_path, spec, is_training=True)
  validation_data = question_answer.DataLoader.from_squad(
      validation_data_path, spec, is_training=False)

  # Fine-tunes the model.
  model = question_answer.create(train_data, model_spec=spec, **kwargs)

  # Gets evaluation results.
  metric = model.evaluate(validation_data)
  tf.compat.v1.logging.info('Eval F1 score:%f' % metric['final_f1'])

  # Exports to TFLite format.
  model.export(export_dir)


def main(_):
  logging.set_verbosity(logging.INFO)

  train_data_path, validation_data_path = download_demo_data()
  run(train_data_path, validation_data_path, FLAGS.export_dir, spec=FLAGS.spec)


if __name__ == '__main__':
  define_flags()
  app.run(main)
