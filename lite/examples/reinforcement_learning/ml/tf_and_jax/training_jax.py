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
"""Experimental JAX/Flax training code for Plane Strike board game."""
import functools
import os
from typing import Sequence
from absl import app
from flax import linen as nn
from flax.metrics import tensorboard
import jax
from jax import random
from jax.experimental import jax2tf
import jax.numpy as jnp
import optax
import tensorflow as tf
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import common
from tqdm import tqdm

ITERATIONS = 500000
LEARNING_RATE = 0.005
MODELDIR = './'
LOGDIR = './jax_log'


class PolicyGradient(nn.Module):
  """Neural network to predict the next strike position."""

  @nn.compact
  def __call__(self, x):
    dtype = jnp.float32
    x = x.reshape((x.shape[0], -1))
    x = nn.Dense(
        features=2 * common.BOARD_SIZE**2, name='hidden1', dtype=dtype)(
            x)
    x = nn.relu(x)
    x = nn.Dense(features=common.BOARD_SIZE**2, name='hidden2', dtype=dtype)(x)
    x = nn.relu(x)
    x = nn.Dense(features=common.BOARD_SIZE**2, name='logits', dtype=dtype)(x)
    policy_probabilities = nn.softmax(x)
    return policy_probabilities


def create_optimizer(learning_rate: float):
  return optax.sgd(learning_rate=learning_rate)


def compute_loss(logits, labels, rewards):
  one_hot_labels = jax.nn.one_hot(labels, num_classes=common.BOARD_SIZE**2)
  loss = -jnp.mean(
      jnp.sum(one_hot_labels * jnp.log(logits), axis=-1) * jnp.asarray(rewards))
  return loss


def train_step(model_optimizer, params, opt_state, game_board_log,
               predicted_action_log, action_result_log):
  """Run one training step."""

  def loss_fn(model_params):
    logits = run_inference(model_params, game_board_log)
    loss = compute_loss(logits, predicted_action_log, action_result_log)
    return loss

  def compute_grads(params):
    return jax.grad(loss_fn)(params)

  grads = compute_grads(params)
  updates, opt_state = model_optimizer.update(grads, opt_state)
  params = optax.apply_updates(params, updates)
  return model_optimizer, params, opt_state


@jax.jit
def run_inference(model_params, board):
  logits = PolicyGradient().apply({'params': model_params}, board)
  return logits


def train_agent(iterations, modeldir, logdir):
  """Train and convert the model."""
  summary_writer = tensorboard.SummaryWriter(logdir)

  rng = random.PRNGKey(0)
  rng, init_rng = random.split(rng)
  policygradient = PolicyGradient()
  params = policygradient.init(
      init_rng, jnp.ones([1, common.BOARD_SIZE, common.BOARD_SIZE]))['params']
  optimizer = create_optimizer(learning_rate=LEARNING_RATE)
  opt_state = optimizer.init(params)

  # Main training loop
  for i in tqdm(range(iterations)):
    predict_fn = functools.partial(run_inference, params)
    board_log, action_log, result_log = common.play_game(predict_fn)
    rewards = common.compute_rewards(result_log)
    summary_writer.scalar('game_length', len(board_log), i)
    optimizer, params, opt_state = train_step(optimizer, params, opt_state,
                                              board_log, action_log, rewards)

    summary_writer.flush()

  summary_writer.close()

  # Convert to tflite model
  model = PolicyGradient()
  jax_predict_fn = lambda input: model.apply({'params': params}, input)

  tf_predict = tf.function(
      jax2tf.convert(jax_predict_fn, enable_xla=False),
      input_signature=[
          tf.TensorSpec(
              shape=[1, common.BOARD_SIZE, common.BOARD_SIZE],
              dtype=tf.float32,
              name='input')
      ],
      autograph=False,
  )

  converter = tf.lite.TFLiteConverter.from_concrete_functions(
      [tf_predict.get_concrete_function()], tf_predict)

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
