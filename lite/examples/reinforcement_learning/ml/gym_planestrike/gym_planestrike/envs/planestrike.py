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
"""OpenAI gym environment for the Plane Strike game.

   Refer to http://gym.openai.com/docs/ for more information.
"""
import random

import gym
from gym import spaces
import numpy as np

PLANE_SIZE = 8

# Plane direction
PLANE_HEADING_RIGHT = 0
PLANE_HEADING_UP = 1
PLANE_HEADING_LEFT = 2
PLANE_HEADING_DOWN = 3

# Rewards for each strike
HIT_REWARD = 1
MISS_REWARD = 0
REPEAT_STRIKE_REWARD = -1

# Hidden board cell status; 'occupied' means it's part of the plane
HIDDEN_BOARD_CELL_OCCUPIED = 1
HIDDEN_BOARD_CELL_UNOCCUPIED = 0

# Visible board cell status
BOARD_CELL_HIT = 1
BOARD_CELL_MISS = -1
BOARD_CELL_UNTRIED = 0


class PlaneStrikeEnv(gym.Env):
  """A class that defines the Plane Strike environement."""

  metadata = {'render.modes': ['human']}

  def init_hidden_board(self):
    hidden_board = np.ones(
        (self.board_size, self.board_size)) * HIDDEN_BOARD_CELL_UNOCCUPIED

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
      plane_core_row = random.randint(1, self.board_size - 2)
      plane_core_column = random.randint(2, self.board_size - 2)
      # Populate the tail
      hidden_board[plane_core_row][plane_core_column -
                                   2] = HIDDEN_BOARD_CELL_OCCUPIED
      hidden_board[plane_core_row - 1][plane_core_column -
                                       2] = HIDDEN_BOARD_CELL_OCCUPIED
      hidden_board[plane_core_row + 1][plane_core_column -
                                       2] = HIDDEN_BOARD_CELL_OCCUPIED
    elif plane_orientation == PLANE_HEADING_UP:
      plane_core_row = random.randint(1, self.board_size - 3)
      plane_core_column = random.randint(1, self.board_size - 3)
      # Populate the tail
      hidden_board[plane_core_row +
                   2][plane_core_column] = HIDDEN_BOARD_CELL_OCCUPIED
      hidden_board[plane_core_row + 2][plane_core_column +
                                       1] = HIDDEN_BOARD_CELL_OCCUPIED
      hidden_board[plane_core_row + 2][plane_core_column -
                                       1] = HIDDEN_BOARD_CELL_OCCUPIED
    elif plane_orientation == PLANE_HEADING_LEFT:
      plane_core_row = random.randint(1, self.board_size - 2)
      plane_core_column = random.randint(1, self.board_size - 3)
      # Populate the tail
      hidden_board[plane_core_row][plane_core_column +
                                   2] = HIDDEN_BOARD_CELL_OCCUPIED
      hidden_board[plane_core_row - 1][plane_core_column +
                                       2] = HIDDEN_BOARD_CELL_OCCUPIED
      hidden_board[plane_core_row + 1][plane_core_column +
                                       2] = HIDDEN_BOARD_CELL_OCCUPIED
    elif plane_orientation == PLANE_HEADING_DOWN:
      plane_core_row = random.randint(2, self.board_size - 2)
      plane_core_column = random.randint(1, self.board_size - 2)
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

  def __init__(self, board_size) -> None:
    super().__init__()
    assert board_size >= 4
    self.board_size = board_size
    self.set_board()

  def step(self, action):
    if self.hit_count == self.plane_size:
      return self.observation, 0, True, {}

    action_x = action // self.board_size
    action_y = action % self.board_size
    # Hit
    if self.hidden_board[action_x][action_y] == HIDDEN_BOARD_CELL_OCCUPIED:
      # Non-repeat move
      if self.observation[action_x][action_y] == BOARD_CELL_UNTRIED:
        self.hit_count = self.hit_count + 1
        self.observation[action_x][action_y] = BOARD_CELL_HIT
        # Successful strike
        if self.hit_count == self.plane_size:
          # Game finished
          return self.observation, HIT_REWARD, True, {}
        else:
          return self.observation, HIT_REWARD, False, {}
      # Repeat strike
      else:
        return self.observation, REPEAT_STRIKE_REWARD, False, {}
    # Miss
    else:
      # Unsuccessful strike
      self.observation[action_x][action_y] = BOARD_CELL_MISS
      return self.observation, MISS_REWARD, False, {}

  def reset(self):
    self.set_board()
    return self.observation

  def render(self, mode='human'):
    print(self.observation)
    return

  def close(self):
    return

  def set_board(self):
    self.plane_size = PLANE_SIZE
    self.hit_count = 0
    self.observation = np.zeros((self.board_size, self.board_size))
    self.hidden_board = self.init_hidden_board()
    self.action_space = spaces.Discrete(self.board_size * self.board_size)
    self.observation_space = spaces.MultiBinary(
        [self.board_size, self.board_size])
