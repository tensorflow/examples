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
import gym
from gym import spaces
import numpy as np
from tensorflow_examples.lite.examples.reinforcement_learning.ml import common


# Rewards for each strike
HIT_REWARD = 1
MISS_REWARD = 0
REPEAT_STRIKE_REWARD = -1


class PlaneStrikeEnv(gym.Env):
  """A class that defines the Plane Strike environement."""

  metadata = {'render.modes': ['human']}

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
    if self.hidden_board[action_x][
        action_y] == common.HIDDEN_BOARD_CELL_OCCUPIED:
      # Non-repeat move
      if self.observation[action_x][action_y] == common.BOARD_CELL_UNTRIED:
        self.hit_count = self.hit_count + 1
        self.observation[action_x][action_y] = common.BOARD_CELL_HIT
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
      self.observation[action_x][action_y] = common.BOARD_CELL_MISS
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
    self.plane_size = common.PLANE_SIZE
    self.hit_count = 0
    self.observation = np.zeros((self.board_size, self.board_size))
    self.hidden_board = common.initialize_random_hidden_board(self.board_size)
    self.action_space = spaces.Discrete(self.board_size * self.board_size)
    self.observation_space = spaces.MultiDiscrete(
        [self.board_size, self.board_size])
