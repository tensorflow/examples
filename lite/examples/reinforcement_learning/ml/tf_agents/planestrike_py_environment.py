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
"""TF Agents python environment for the PlaneStrike board game."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import common
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

# We always use square board, so only one size is needed
BOARD_SIZE = 8

MAX_STEPS_PER_EPISODE = BOARD_SIZE**2

# Plane direction
PLANE_HEADING_RIGHT = 0
PLANE_HEADING_UP = 1
PLANE_HEADING_LEFT = 2
PLANE_HEADING_DOWN = 3

# Rewards for each strike
HIT_REWARD = 1
MISS_REWARD = 0
REPEAT_STRIKE_REWARD = -1
# Reward for finishing the game within MAX_STEPS_PER_EPISODE
FINISHED_GAME_REWARD = 10
# Reward for not finishing the game within MAX_STEPS_PER_EPISODE
UNFINISHED_GAME_REWARD = -10

# Hidden board cell status; 'occupied' means it's part of the plane
HIDDEN_BOARD_CELL_OCCUPIED = 1
HIDDEN_BOARD_CELL_UNOCCUPIED = 0

# Visible board cell status
VISIBLE_BOARD_CELL_HIT = 1
VISIBLE_BOARD_CELL_MISS = -1
VISIBLE_BOARD_CELL_UNTRIED = 0


class PlaneStrikePyEnvironment(py_environment.PyEnvironment):
  """PlaneStrike environment for TF Agents."""

  def __init__(self,
               board_size=BOARD_SIZE,
               discount=0.9,
               max_steps=MAX_STEPS_PER_EPISODE) -> None:
    super(PlaneStrikePyEnvironment, self).__init__()
    assert board_size >= 4
    self._board_size = board_size
    self._strike_count = 0
    self._discount = discount
    self._max_steps = max_steps
    self._episode_ended = False
    self._action_spec = array_spec.BoundedArraySpec(
        (), np.int32, minimum=0, maximum=self._board_size**2 - 1)
    self._observation_spec = array_spec.BoundedArraySpec(
        (self._board_size, self._board_size),
        np.float32,
        minimum=VISIBLE_BOARD_CELL_MISS,
        maximum=VISIBLE_BOARD_CELL_HIT)
    self._time_step_spec = ts.time_step_spec(self._observation_spec)
    self.set_boards()

  def set_boards(self):
    self._plane_size = common.PLANE_SIZE
    self._hit_count = 0
    self._visible_board = np.zeros((self._board_size, self._board_size))
    self._hidden_board = common.initialize_random_hidden_board(self._board_size)

  def current_time_step(self):
    return self._current_time_step

  def observation_spec(self):
    """Return observation_spec."""
    return self._observation_spec

  def action_spec(self):
    """Return action_spec."""
    return self._action_spec

  def _reset(self):
    """Return initial_time_step."""
    self._episode_ended = False
    self._strike_count = 0
    self._hit_count = 0
    self.set_boards()
    return ts.restart(np.array(self._visible_board, dtype=np.float32))

  def _step(self, action):
    """Apply action and return new time_step."""
    if self._hit_count == self._plane_size:
      self._episode_ended = True
      return self.reset()

    if self._strike_count + 1 == self._max_steps:
      self.reset()
      return ts.termination(
          np.array(self._visible_board, dtype=np.float32),
          UNFINISHED_GAME_REWARD)

    self._strike_count += 1
    action_x = action // self._board_size
    action_y = action % self._board_size
    # Hit
    if self._hidden_board[action_x][action_y] == HIDDEN_BOARD_CELL_OCCUPIED:
      # Non-repeat move
      if self._visible_board[action_x][action_y] == VISIBLE_BOARD_CELL_UNTRIED:
        self._hit_count += 1
        self._visible_board[action_x][action_y] = VISIBLE_BOARD_CELL_HIT
        # Successful strike
        if self._hit_count == self._plane_size:
          # Game finished
          self._episode_ended = True
          return ts.termination(
              np.array(self._visible_board, dtype=np.float32),
              FINISHED_GAME_REWARD)
        else:
          self._episode_ended = False
          return ts.transition(
              np.array(self._visible_board, dtype=np.float32), HIT_REWARD,
              self._discount)
      # Repeat strike
      else:
        self._episode_ended = False
        return ts.transition(
            np.array(self._visible_board, dtype=np.float32),
            REPEAT_STRIKE_REWARD, self._discount)
    # Miss
    else:
      # Unsuccessful strike
      self._episode_ended = False
      self._visible_board[action_x][action_y] = VISIBLE_BOARD_CELL_MISS
      return ts.transition(
          np.array(self._visible_board, dtype=np.float32), MISS_REWARD,
          self._discount)

  def render(self, mode: "human") -> np.ndarray:
    if mode != "human":
      raise ValueError(
          "Only rendering mode supported is 'human', got {} instead.".format(
              mode))
    return self._visible_board
