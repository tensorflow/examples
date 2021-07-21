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
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * The class that implements a policy gradient agent to play the game, assuming model is trained
 * using TensorFlow or JAX.
 */
public class RLAgent extends PlaneStrikeAgent {

  private ByteBuffer boardData = null;
  private final float[][] outputProbArrays =
      new float[1][Constants.BOARD_SIZE * Constants.BOARD_SIZE];

  public RLAgent(Activity activity) throws IOException {
    super(activity);

    boardData = ByteBuffer.allocateDirect(Constants.BOARD_SIZE * Constants.BOARD_SIZE * 4);
    boardData.order(ByteOrder.nativeOrder());
  }

  /** Predict the next move based on current board state. */
  @Override
  protected StrikePrediction predictNextMove(BoardCellStatus[][] board) {

    if (tflite == null) {
      Log.e(
          Constants.TAG, "Game agent failed to initialize. Please restart the app.");
      return null;
    } else {
      prepareModelInput(board);
      runInference();
    }

    StrikePrediction strikePrediction = new StrikePrediction();
    strikePrediction.x = agentStrikePosition / Constants.BOARD_SIZE;
    strikePrediction.y = agentStrikePosition % Constants.BOARD_SIZE;
    return strikePrediction;
  }

  /** Run model inference on current board state. */
  @Override
  protected void runInference() {
    tflite.run(boardData, outputProbArrays);
    boardData.rewind();

    float[] probArray = outputProbArrays[0]; // batch size is 1 so we use [0] here
    // Argmax
    int maxIndex = 0;
    for (int i = 0; i < probArray.length; i++) {
      maxIndex = probArray[i] > probArray[maxIndex] ? i : maxIndex;
    }
    agentStrikePosition = maxIndex;
    isPredictedByAgent = true;
  }

  @Override
  protected void prepareModelInput(BoardCellStatus[][] board) {
    if (board == null) {
      return;
    }
    float boardCellStatusValue = 0;
    for (int i = 0; i < Constants.BOARD_SIZE; ++i) {
      for (int j = 0; j < Constants.BOARD_SIZE; ++j) {
        switch (board[i][j]) {
          case HIT:
            boardCellStatusValue = Constants.CELL_STATUS_VALUE_HIT;
            break;
          case MISS:
            boardCellStatusValue = Constants.CELL_STATUS_VALUE_MISS;
            break;
          default:
            boardCellStatusValue = Constants.CELL_STATUS_VALUE_UNTRIED;
            break;
        }
        boardData.putFloat(boardCellStatusValue);
      }
    }
  }
}
