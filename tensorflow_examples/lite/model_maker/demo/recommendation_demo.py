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
"""Recommendation demo code of Model Maker for TFLite."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app
from absl import flags
from absl import logging

from tensorflow_examples.lite.model_maker.core.data_util.recommendation_dataloader import RecommendationDataLoader
from tensorflow_examples.lite.model_maker.core.task import recommendation

FLAGS = flags.FLAGS


def define_flags():
  flags.DEFINE_string('data_dir', None, 'The directory to save dataset.')
  flags.DEFINE_string('export_dir', None,
                      'The directory to save exported files.')
  flags.DEFINE_string('spec', 'recommendation_bow',
                      'The recommendation model to run.')
  flags.mark_flag_as_required('export_dir')


def download_data(download_dir):
  """Downloads demo data, and returns directory path."""
  return RecommendationDataLoader.download_and_extract_movielens(download_dir)


def run(data_dir,
        export_dir,
        spec='recommendation_bow',
        batch_size=16,
        epochs=5):
  """Runs demo."""
  train_data = RecommendationDataLoader.from_movielens(data_dir, 'train')
  test_data = RecommendationDataLoader.from_movielens(data_dir, 'test')

  # options for spec to specify recommendation model architecture.
  model_spec_options = dict(
      context_embedding_dim=16,
      label_embedding_dim=16,
      item_vocab_size=train_data.max_vocab_id,
      hidden_layer_dim_ratios=[1, 1],
  )
  # Create a model and train.
  model = recommendation.create(
      train_data,
      model_spec=spec,
      model_spec_options=model_spec_options,
      model_dir=export_dir,
      validation_data=test_data,
      batch_size=batch_size,
      epochs=epochs)

  # Evaluate with test_data.
  history = model.evaluate(test_data)
  print('Test metrics from Keras model: %s' % history)

  # Export tflite model.
  model.export(export_dir)

  # Evaluate tflite model.
  tflite_model = os.path.join(export_dir, 'model.tflite')
  history = model.evaluate_tflite(tflite_model, test_data)
  print('Test metrics from TFLite model: %s' % history)


def main(_):
  logging.set_verbosity(logging.INFO)
  extracted_dir = download_data(FLAGS.data_dir)
  run(extracted_dir, FLAGS.export_dir, spec=FLAGS.spec)


if __name__ == '__main__':
  define_flags()
  app.run(main)
