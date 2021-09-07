# Lint as: python3
#   Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""TensorFlow training code for Plane Strike board game."""
import os
from typing import Sequence
from absl import app
import tensorflow as tf
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import common

ITERATIONS = 80000
LEARNING_RATE = 0.002
MODELDIR = './'
LOGDIR = './tf_log'


def train_agent(iterations, modeldir, logdir):
  """Train and convert the model."""

  model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(
          input_shape=(common.BOARD_SIZE, common.BOARD_SIZE)),
      tf.keras.layers.Dense(2 * common.BOARD_SIZE**2, activation='relu'),
      tf.keras.layers.Dense(common.BOARD_SIZE**2, activation='relu'),
      tf.keras.layers.Dense(common.BOARD_SIZE**2, activation='softmax')
  ])

  sgd = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE)

  model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd)

  summary_writer = tf.summary.create_file_writer(logdir)

  def predict_fn(board):
    return model.predict(board)

  # Main training loop
  progress_bar = tf.keras.utils.Progbar(iterations)
  for i in range(iterations):
    board_log, action_log, result_log = common.play_game(predict_fn)
    with summary_writer.as_default():
      tf.summary.scalar('game_length', len(action_log), step=i)
    rewards = common.compute_rewards(result_log)

    model.fit(
        x=board_log,
        y=action_log,
        batch_size=1,
        verbose=0,
        epochs=1,
        sample_weight=rewards)

    summary_writer.flush()
    progress_bar.add(1)

  summary_writer.close()

  # Convert to tflite model
  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  tflite_model = converter.convert()

  # Save the model
  with open(os.path.join(modeldir, 'planestrike.tflite'), 'wb') as f:
    f.write(tflite_model)

  print('TFLite model generated!')


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  train_agent(ITERATIONS, MODELDIR, LOGDIR)


if __name__ == '__main__':
  app.run(main)
