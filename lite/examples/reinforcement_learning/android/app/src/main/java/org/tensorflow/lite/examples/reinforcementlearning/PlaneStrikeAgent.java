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
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import org.tensorflow.lite.Interpreter;

/** The class that defines a policy gradient agent to play the game. */
public abstract class PlaneStrikeAgent {

  protected Interpreter tflite;
  protected Interpreter.Options tfliteOptions;

  protected int agentStrikePosition;

  public PlaneStrikeAgent(Activity activity) throws IOException {
    tfliteOptions = new Interpreter.Options();
    tflite = new Interpreter(loadModelFile(activity), tfliteOptions);
  }

  /** Predict the next move based on current board state. */
  protected abstract int predictNextMove(BoardCellStatus[][] board);

  /** Run model inference on current board state. */
  protected abstract void runInference();

  protected abstract void prepareModelInput(BoardCellStatus[][] board);

  /** Memory-map the model file in Assets. */
  protected MappedByteBuffer loadModelFile(Activity activity) throws IOException {

    String model;
    if (Constants.USE_MODEL_FROM_TF) {
      model = Constants.TF_TFLITE_MODEL;
    } else {
      model = Constants.TF_AGENTS_TFLITE_MODEL;
    }
    AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(model);
    FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
    FileChannel fileChannel = inputStream.getChannel();
    long startOffset = fileDescriptor.getStartOffset();
    long declaredLength = fileDescriptor.getDeclaredLength();
    return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
  }
}
