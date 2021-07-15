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
import android.content.res.AssetFileDescriptor;
import android.util.Log;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Random;
import org.tensorflow.lite.Interpreter;

/** The class that implements a policy gradient agent to play the game. */
public class PolicyGradientAgent {

  private final Interpreter tflite;
  private final Interpreter.Options tfliteOptions;
  private ByteBuffer boardData = null;
  private final float[][] outputProbArrays =
      new float[1][Constants.BOARD_SIZE * Constants.BOARD_SIZE];

  PolicyGradientAgent(Activity activity) throws IOException {
    tfliteOptions = new Interpreter.Options();
    tflite = new Interpreter(loadModelFile(activity), tfliteOptions);
    // a 32bit float value requires 4 bytes
    boardData = ByteBuffer.allocateDirect(Constants.BOARD_SIZE * Constants.BOARD_SIZE * 4);
    boardData.order(ByteOrder.nativeOrder());
  }

  /** Predict the next move based on current board state. */
  StrikePrediction predictNextMove(BoardCellStatus[][] board) {
    int agentStrikePosition;
    boolean isPredictedByAgent;
    if (tflite == null) {
      Log.e(Constants.TAG, "Policy gradient agent has not been initialized; skipping ...");
      Random random = new Random();
      agentStrikePosition = random.nextInt(Constants.BOARD_SIZE * Constants.BOARD_SIZE);
      isPredictedByAgent = false;
    } else {
      convertBoardStateToByteBuffer(board);
      runInference();
      float[] probArray = outputProbArrays[0]; // batch size is 1 so we use [0] here
      // Argmax
      int maxIndex = 0;
      for (int i = 0; i < probArray.length; i++) {
        maxIndex = probArray[i] > probArray[maxIndex] ? i : maxIndex;
      }
      agentStrikePosition = maxIndex;
      isPredictedByAgent = true;
    }

    StrikePrediction strikePrediction = new StrikePrediction();
    strikePrediction.x = agentStrikePosition / Constants.BOARD_SIZE;
    strikePrediction.y = agentStrikePosition % Constants.BOARD_SIZE;
    strikePrediction.isPredictedByAgent = isPredictedByAgent;
    return strikePrediction;
  }

  /** Run model inference on current board state. */
  private void runInference() {
    tflite.run(boardData, outputProbArrays);
    boardData.rewind();
  }

  private void convertBoardStateToByteBuffer(BoardCellStatus[][] board) {
    if (board == null) {
      return;
    }
    float boardCellStatusValue = 0;
    for (int i = 0; i < Constants.BOARD_SIZE; ++i) {
      for (int j = 0; j < Constants.BOARD_SIZE; ++j) {
        switch (board[i][j]) {
          case HIT:
            boardCellStatusValue = 1;
            break;
          case MISS:
            boardCellStatusValue = -1;
            break;
          default:
            boardCellStatusValue = 0;
            break;
        }
        boardData.putFloat(boardCellStatusValue);
      }
    }
  }

  /** Memory-map the model file in Assets. */
  private MappedByteBuffer loadModelFile(Activity activity) throws IOException {
    AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(Constants.TFLITE_MODEL);
    FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
    FileChannel fileChannel = inputStream.getChannel();
    long startOffset = fileDescriptor.getStartOffset();
    long declaredLength = fileDescriptor.getDeclaredLength();
    return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
  }
}
