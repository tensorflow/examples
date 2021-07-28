/*
 * Copyright 2020 The TensorFlow Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.reinforcementlearning;

import android.app.Activity;
import android.util.Log;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/**
 * The class that implements an agent to play the game, assuming model is trained
 * using TensorFlow Agents REINFORCE agent.
 */
public class RLAgentFromTFAgents extends PlaneStrikeAgent {

  private final Object[] inputs = new Object[4];

  public RLAgentFromTFAgents(Activity activity) throws IOException {
    super(activity);
  }

  /** Predict the next move based on current board state. */
  @Override
  protected int predictNextMove(BoardCellStatus[][] board) {

    if (tflite == null) {
      Log.e(
          Constants.TAG, "Game agent failed to initialize. Please restart the app.");
      return -1;
    } else {
      prepareModelInput(board);
      runInference();
    }

    return agentStrikePosition;
  }

  /** Run model inference on current board state. */
  @Override
  protected void runInference() {
    Map<Integer, Object> output = new HashMap<>();
    // TF Agent directly returns the predicted action
    int[][] prediction = new int[1][1];
    output.put(0, prediction);
    tflite.runForMultipleInputsOutputs(inputs, output);
    agentStrikePosition = prediction[0][0];
  }

  @Override
  protected void prepareModelInput(BoardCellStatus[][] board) {
    if (board == null) {
      return;
    }

    // Model converted from TF Agents takes 4 tensors as input; only the 3rd one 'observation' is
    // useful for inference
    int stepType = 0;
    float discount = 0;
    float reward = 0;
    inputs[0] = stepType;
    inputs[1] = discount;
    float[][][] boardState = new float[1][8][8];
    for (int i = 0; i < Constants.BOARD_SIZE; ++i) {
      for (int j = 0; j < Constants.BOARD_SIZE; ++j) {
        switch (board[i][j]) {
          case HIT:
            boardState[0][i][j] = Constants.CELL_STATUS_VALUE_HIT;
            break;
          case MISS:
            boardState[0][i][j] = Constants.CELL_STATUS_VALUE_MISS;
            break;
          default:
            boardState[0][i][j] = Constants.CELL_STATUS_VALUE_UNTRIED;
            break;
        }
      }
    }
    inputs[2] = boardState;
    inputs[3] = reward;
  }
}
