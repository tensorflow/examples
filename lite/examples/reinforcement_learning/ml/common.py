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
"""Common utils for both TF and JAX training."""

import gym
# pylint: disable=unused-import
import gym_planestrike
import numpy as np

# We always use square board, so only one size is needed
BOARD_SIZE = 8
PLANE_SIZE = 8

# Reward discount factor
GAMMA = 0.5


def play_game(predict_fn):
  """Play one round of game to gather logs."""

  env = gym.make('PlaneStrike-v0', board_size=BOARD_SIZE)
  observation = env.reset()

  game_board_log = []
  predicted_action_log = []
  action_result_log = []
  while True:
    probs = predict_fn(np.expand_dims(observation, 0))[0]
    probs = [
        p * (index not in predicted_action_log) for index, p in enumerate(probs)
    ]
    probs = [p / sum(probs) for p in probs]
    assert sum(probs) > 0.999999
    game_board_log.append(np.copy(observation))
    strike_pos = np.random.choice(BOARD_SIZE**2, p=probs)
    observation, reward, done, _ = env.step(strike_pos)
    action_result_log.append(reward)
    predicted_action_log.append(strike_pos)
    if done:
      env.close()
      return np.asarray(game_board_log), np.asarray(
          predicted_action_log), np.asarray(action_result_log)


def compute_rewards(game_result_log, gamma=GAMMA):
  """Compute discounted rewards."""
  discounted_rewards = []
  discounted_sum = 0

  for reward in game_result_log[::-1]:
    discounted_sum = reward + gamma * discounted_sum
    discounted_rewards.append(discounted_sum)
  return np.asarray(discounted_rewards[::-1])
