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

from tflite_model_maker import model_spec as ms
from tflite_model_maker import recommendation

FLAGS = flags.FLAGS


def define_flags():
  flags.DEFINE_string('data_dir', None, 'The directory to save dataset.')
  flags.DEFINE_string('export_dir', None,
                      'The directory to save exported files.')
  flags.DEFINE_string('encoder_type', 'bow',
                      'The recommendation encoder to run. (bow, cnn, lstm)')
  flags.mark_flag_as_required('export_dir')


def download_data(download_dir):
  """Downloads demo data, and returns directory path."""
  return recommendation.DataLoader.download_and_extract_movielens(download_dir)


def get_input_spec(encoder_type: str,
                   num_classes: int) -> recommendation.spec.InputSpec:
  """Gets input spec (for test).

  Input spec defines how the input features are extracted.

  Args:
    encoder_type: str, case-insensitive {'CNN', 'LSTM', 'BOW'}.
    num_classes: int, num of classes in vocabulary.

  Returns:
    InputSpec.
  """
  etype = encoder_type.upper()
  if etype not in {'CNN', 'LSTM', 'BOW'}:
    raise ValueError('Not support encoder_type: {}'.format(etype))

  return recommendation.spec.InputSpec(
      activity_feature_groups=[
          # Group #1: defines how features are grouped in the first Group.
          dict(
              features=[
                  # First feature.
                  dict(
                      feature_name='context_movie_id',  # Feature name
                      feature_type='INT',  # Feature type
                      vocab_size=num_classes,  # ID size (number of IDs)
                      embedding_dim=8,  # Projected feature embedding dim
                      feature_length=10,  # History length of 10.
                  ),
                  # Maybe more features...
              ],
              encoder_type='CNN',  # CNN encoder (e.g. CNN, LSTM, BOW)
          ),
          # Maybe more groups...
      ],
      label_feature=dict(
          feature_name='label_movie_id',  # Label feature name
          feature_type='INT',  # Label type
          vocab_size=num_classes,  # Label size (number of classes)
          embedding_dim=8,  # Label embedding demension
          feature_length=1,  # Exactly 1 label
      ),
  )


def get_model_hparams() -> recommendation.spec.ModelHParams:
  """Gets model hparams (for test).

  ModelHParams defines the model architecture.

  Returns:
    ModelHParams.
  """
  return recommendation.spec.ModelHParams(
      hidden_layer_dims=[32, 32],  # Hidden layers dimension.
      eval_top_k=[1, 5],  # Eval top 1 and top 5.
      conv_num_filter_ratios=[2, 4],  # For CNN encoder, conv filter mutipler.
      conv_kernel_size=16,  # For CNN encoder, base kernel size.
      lstm_num_units=16,  # For LSTM/RNN, num units.
      num_predictions=10,  # Number of output predictions. Select top 10.
  )


def run(data_dir, export_dir, batch_size=16, epochs=5, encoder_type='bow'):
  """Runs demo."""
  meta = recommendation.DataLoader.generate_movielens_dataset(data_dir)
  num_classes = recommendation.DataLoader.get_num_classes(meta)

  input_spec = get_input_spec(encoder_type, num_classes)
  train_data = recommendation.DataLoader.from_movielens(data_dir, 'train',
                                                        input_spec)
  test_data = recommendation.DataLoader.from_movielens(data_dir, 'test',
                                                       input_spec)

  model_spec = ms.get(
      'recommendation',
      input_spec=input_spec,
      model_hparams=get_model_hparams())
  # Create a model and train.
  model = recommendation.create(
      train_data,
      model_spec=model_spec,
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

  export_dir = os.path.expanduser(FLAGS.export_dir)
  data_dir = os.path.expanduser(FLAGS.data_dir)

  extracted_dir = download_data(data_dir)
  run(extracted_dir, export_dir, encoder_type=FLAGS.encoder_type)


if __name__ == '__main__':
  define_flags()
  app.run(main)
