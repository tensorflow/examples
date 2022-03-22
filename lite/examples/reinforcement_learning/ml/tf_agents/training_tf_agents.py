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
"""TF Agents training code for the Plane Strike board game."""
import os
from typing import Sequence

from absl import app
import reverb
import tensorflow as tf
import planestrike_py_environment
import tensorflow_probability as tfp
import tf_agents as tfa
from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.drivers import py_driver
from tf_agents.environments import tf_py_environment
from tf_agents.policies import policy_saver
from tf_agents.policies import py_tf_eager_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.specs import tensor_spec
from tf_agents.utils import common

BOARD_SIZE = 8
ITERATIONS = 250000
COLLECT_EPISODES_PER_ITERATION = 1
REPLAY_BUFFER_CAPACITY = 2000
REPLAY_BUFFER_TABLE_NAME = 'uniform_table'
DISCOUNT = 0.5

FC_LAYER_PARAMS = 100

LEARNING_RATE = 1e-3
NUM_EVAL_EPISODES = 20
EVAL_INTERVAL = 500

LOGDIR = './tf_agents_log'
MODELDIR = './'
POLICYDIR = './policy'


def compute_avg_return_and_steps(environment, policy, num_episodes=10):
  """Compute average return and # of steps."""
  total_return = 0.0
  total_steps = 0.0
  for _ in range(num_episodes):

    time_step = environment.reset()
    episode_return = 0.0
    episode_steps = 0.0

    while not time_step.is_last():
      action_step = policy.action(time_step)
      time_step = environment.step(action_step.action)
      episode_return += time_step.reward
      episode_steps += 1
    total_return += episode_return
    total_steps += episode_steps

  average_return = total_return / num_episodes
  average_episode_steps = total_steps / num_episodes
  return average_return.numpy()[0], average_episode_steps


def collect_episode(environment, policy, num_episodes, replay_buffer_observer):
  """Collect game episode trajectories."""
  initial_time_step = environment.reset()

  driver = py_driver.PyDriver(
      environment,
      py_tf_eager_policy.PyTFEagerPolicy(policy, use_tf_function=True),
      [replay_buffer_observer],
      max_episodes=num_episodes)
  initial_time_step = environment.reset()
  driver.run(initial_time_step)


def train_agent(iterations, modeldir, logdir, policydir):
  """Train and convert the model using TF Agents."""

  train_py_env = planestrike_py_environment.PlaneStrikePyEnvironment(
      board_size=BOARD_SIZE, discount=DISCOUNT, max_steps=BOARD_SIZE**2)
  eval_py_env = planestrike_py_environment.PlaneStrikePyEnvironment(
      board_size=BOARD_SIZE, discount=DISCOUNT, max_steps=BOARD_SIZE**2)

  train_env = tf_py_environment.TFPyEnvironment(train_py_env)
  eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

  # Alternatively you could use ActorDistributionNetwork as actor_net
  actor_net = tfa.networks.Sequential([
      tfa.keras_layers.InnerReshape([BOARD_SIZE, BOARD_SIZE], [BOARD_SIZE**2]),
      tf.keras.layers.Dense(FC_LAYER_PARAMS, activation='relu'),
      tf.keras.layers.Dense(BOARD_SIZE**2),
      tf.keras.layers.Lambda(lambda t: tfp.distributions.Categorical(logits=t)),
  ],
                                      input_spec=train_py_env.observation_spec(
                                      ))

  optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

  train_step_counter = tf.Variable(0)

  tf_agent = reinforce_agent.ReinforceAgent(
      train_env.time_step_spec(),
      train_env.action_spec(),
      actor_network=actor_net,
      optimizer=optimizer,
      normalize_returns=True,
      train_step_counter=train_step_counter)

  tf_agent.initialize()

  eval_policy = tf_agent.policy
  collect_policy = tf_agent.collect_policy

  tf_policy_saver = policy_saver.PolicySaver(collect_policy)

  # Use reverb as replay buffer
  replay_buffer_signature = tensor_spec.from_spec(tf_agent.collect_data_spec)
  table = reverb.Table(
      REPLAY_BUFFER_TABLE_NAME,
      max_size=REPLAY_BUFFER_CAPACITY,
      sampler=reverb.selectors.Uniform(),
      remover=reverb.selectors.Fifo(),
      rate_limiter=reverb.rate_limiters.MinSize(1),
      signature=replay_buffer_signature
  )  # specify signature here for validation at insertion time

  reverb_server = reverb.Server([table])

  replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
      tf_agent.collect_data_spec,
      sequence_length=None,
      table_name=REPLAY_BUFFER_TABLE_NAME,
      local_server=reverb_server)

  replay_buffer_observer = reverb_utils.ReverbAddEpisodeObserver(
      replay_buffer.py_client, REPLAY_BUFFER_TABLE_NAME, REPLAY_BUFFER_CAPACITY)

  # Optimize by wrapping some of the code in a graph using TF function.
  tf_agent.train = common.function(tf_agent.train)

  # Evaluate the agent's policy once before training.
  avg_return = compute_avg_return_and_steps(eval_env, tf_agent.policy,
                                            NUM_EVAL_EPISODES)

  summary_writer = tf.summary.create_file_writer(logdir)

  for i in range(iterations):
    # Collect a few episodes using collect_policy and save to the replay buffer.
    collect_episode(train_py_env, collect_policy,
                    COLLECT_EPISODES_PER_ITERATION, replay_buffer_observer)

    # Use data from the buffer and update the agent's network.
    iterator = iter(replay_buffer.as_dataset(sample_batch_size=1))
    trajectories, _ = next(iterator)
    tf_agent.train(experience=trajectories)
    replay_buffer.clear()

    logger = tf.get_logger()
    if i % EVAL_INTERVAL == 0:
      avg_return, avg_episode_length = compute_avg_return_and_steps(
          eval_env, eval_policy, NUM_EVAL_EPISODES)
      with summary_writer.as_default():
        tf.summary.scalar('Average return', avg_return, step=i)
        tf.summary.scalar('Average episode length', avg_episode_length, step=i)
        summary_writer.flush()
      logger.info(
          'iteration = {0}: Average Return = {1}, Average Episode Length = {2}'
          .format(i, avg_return, avg_episode_length))

  summary_writer.close()

  tf_policy_saver.save(policydir)
  # Convert to tflite model
  converter = tf.lite.TFLiteConverter.from_saved_model(
      policydir, signature_keys=['action'])
  converter.target_spec.supported_ops = [
      tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
      tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
  ]
  tflite_policy = converter.convert()
  with open(os.path.join(modeldir, 'planestrike_tf_agents.tflite'), 'wb') as f:
    f.write(tflite_policy)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  train_agent(ITERATIONS, MODELDIR, LOGDIR, POLICYDIR)


if __name__ == '__main__':
  app.run(main)
