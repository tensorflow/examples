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
"""Common utils for training."""

import random

import numpy as np

# We always use square board, so only one size is needed
BOARD_SIZE = 8
PLANE_SIZE = 8

# Reward discount factor
GAMMA = 0.5

# Plane direction
PLANE_HEADING_RIGHT = 0
PLANE_HEADING_UP = 1
PLANE_HEADING_LEFT = 2
PLANE_HEADING_DOWN = 3

# Hidden board cell status; 'occupied' means it's part of the plane
HIDDEN_BOARD_CELL_OCCUPIED = 1
HIDDEN_BOARD_CELL_UNOCCUPIED = 0

# Visible board cell status
BOARD_CELL_HIT = 1
BOARD_CELL_MISS = -1
BOARD_CELL_UNTRIED = 0


def play_game(predict_fn):
  """Play one round of game to gather logs for TF/JAX training."""

  # Only import gym-related libraries when absolutely needed
  # pylint: disable=g-import-not-at-top
  import gym
  # pylint: disable=unused-import,g-import-not-at-top
  import gym_planestrike
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
  """Compute discounted rewards for TF/JAX training."""
  discounted_rewards = []
  discounted_sum = 0

  for reward in game_result_log[::-1]:
    discounted_sum = reward + gamma * discounted_sum
    discounted_rewards.append(discounted_sum)
  return np.asarray(discounted_rewards[::-1])


def initialize_random_hidden_board(board_size):
  """Initialize the hidden board."""

  hidden_board = np.ones(
      (board_size, board_size)) * HIDDEN_BOARD_CELL_UNOCCUPIED

  # Populate the plane's position
  # First figure out the plane's orientation
  #   0: heading right
  #   1: heading up
  #   2: heading left
  #   3: heading down

  plane_orientation = random.randint(0, 3)

  # Figrue out the location of plane core as the '*' below
  #   | |      |      | |    ---
  #   |-*-    -*-    -*-|     |
  #   | |      |      | |    -*-
  #           ---             |
  if plane_orientation == PLANE_HEADING_RIGHT:
    plane_core_row = random.randint(1, board_size - 2)
    plane_core_column = random.randint(2, board_size - 2)
    # Populate the tail
    hidden_board[plane_core_row][plane_core_column -
                                 2] = HIDDEN_BOARD_CELL_OCCUPIED
    hidden_board[plane_core_row - 1][plane_core_column -
                                     2] = HIDDEN_BOARD_CELL_OCCUPIED
    hidden_board[plane_core_row + 1][plane_core_column -
                                     2] = HIDDEN_BOARD_CELL_OCCUPIED
  elif plane_orientation == PLANE_HEADING_UP:
    plane_core_row = random.randint(1, board_size - 3)
    plane_core_column = random.randint(1, board_size - 3)
    # Populate the tail
    hidden_board[plane_core_row +
                 2][plane_core_column] = HIDDEN_BOARD_CELL_OCCUPIED
    hidden_board[plane_core_row + 2][plane_core_column +
                                     1] = HIDDEN_BOARD_CELL_OCCUPIED
    hidden_board[plane_core_row + 2][plane_core_column -
                                     1] = HIDDEN_BOARD_CELL_OCCUPIED
  elif plane_orientation == PLANE_HEADING_LEFT:
    plane_core_row = random.randint(1, board_size - 2)
    plane_core_column = random.randint(1, board_size - 3)
    # Populate the tail
    hidden_board[plane_core_row][plane_core_column +
                                 2] = HIDDEN_BOARD_CELL_OCCUPIED
    hidden_board[plane_core_row - 1][plane_core_column +
                                     2] = HIDDEN_BOARD_CELL_OCCUPIED
    hidden_board[plane_core_row + 1][plane_core_column +
                                     2] = HIDDEN_BOARD_CELL_OCCUPIED
  elif plane_orientation == PLANE_HEADING_DOWN:
    plane_core_row = random.randint(2, board_size - 2)
    plane_core_column = random.randint(1, board_size - 2)
    # Populate the tail
    hidden_board[plane_core_row -
                 2][plane_core_column] = HIDDEN_BOARD_CELL_OCCUPIED
    hidden_board[plane_core_row - 2][plane_core_column +
                                     1] = HIDDEN_BOARD_CELL_OCCUPIED
    hidden_board[plane_core_row - 2][plane_core_column -
                                     1] = HIDDEN_BOARD_CELL_OCCUPIED

  # Populate the cross
  hidden_board[plane_core_row][plane_core_column] = HIDDEN_BOARD_CELL_OCCUPIED
  hidden_board[plane_core_row +
               1][plane_core_column] = HIDDEN_BOARD_CELL_OCCUPIED
  hidden_board[plane_core_row -
               1][plane_core_column] = HIDDEN_BOARD_CELL_OCCUPIED
  hidden_board[plane_core_row][plane_core_column +
                               1] = HIDDEN_BOARD_CELL_OCCUPIED
  hidden_board[plane_core_row][plane_core_column -
                               1] = HIDDEN_BOARD_CELL_OCCUPIED

  return hidden_board
