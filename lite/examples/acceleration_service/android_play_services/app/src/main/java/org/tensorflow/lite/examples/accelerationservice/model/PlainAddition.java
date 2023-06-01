/*
 * Copyright 2023 The TensorFlow Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.accelerationservice.model;

import android.content.Context;
import com.google.android.gms.tflite.acceleration.Model;
import com.google.android.gms.tflite.acceleration.Model.ModelLocation;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import org.tensorflow.lite.examples.accelerationservice.logger.Logger;
import org.tensorflow.lite.support.common.FileUtil;

final class PlainAddition implements AssetModel {

  private static final String MODEL_ID = "add";
  private static final String MODEL_NAMESPACE = "tflite";
  private static final String MODEL_PATH = "add.tflite";

  private static final int BATCH_SIZE = 5;
  private static final int INPUT_TENSOR_SIZE = 1;

  private static final int MODEL_INPUT_DIMENSIONS = product(BATCH_SIZE, 8, 8, 3);
  private static final int MODEL_OUTPUT_DIMENSIONS = product(BATCH_SIZE, 8, 8, 3);

  private final Model model;
  private final Logger logger;

  public PlainAddition(Context context, Logger logger) throws IOException {
    ByteBuffer modelBuffer = FileUtil.loadMappedFile(context, MODEL_PATH);
    this.model =
        new Model.Builder()
            .setModelId(MODEL_ID)
            .setModelNamespace(MODEL_NAMESPACE)
            .setModelLocation(ModelLocation.fromByteBuffer(modelBuffer))
            .build();
    this.logger = logger;
  }

  @Override
  public Model getModel() {
    return model;
  }

  @Override
  public int getBatchSize() {
    return BATCH_SIZE;
  }

  @Override
  public Object[] getInputs() {
    Object[] batchedInputs = new Object[INPUT_TENSOR_SIZE];
    for (int i = 0; i < batchedInputs.length; i++) {
      float[] batchedInput = new float[MODEL_INPUT_DIMENSIONS];
      for (int j = 0; j < batchedInput.length; j++) {
        batchedInput[j] = generateInput(j);
      }
      batchedInputs[i] = batchedInput;
    }
    return batchedInputs;
  }

  @Override
  public Map<Integer, Object> allocateOutputs() {
    Map<Integer, Object> outputs = new HashMap<>();
    for (int i = 0; i < INPUT_TENSOR_SIZE; i++) {
      outputs.put(i, new float[MODEL_OUTPUT_DIMENSIONS]);
    }
    return outputs;
  }

  @Override
  public boolean validateBenchmarkOutputs(ByteBuffer[] outputs) {
    Object[] floatOutputs = new Object[outputs.length];
    if (outputs.length != INPUT_TENSOR_SIZE) {
      return false;
    }
    for (int i = 0; i < outputs.length; i++) {
      floatOutputs[i] = toFloatArray(outputs[i]);
    }
    return validateInterpreterOutputs(floatOutputs);
  }

  @Override
  public boolean validateInterpreterOutputs(Object[] outputs) {
    if (outputs.length != INPUT_TENSOR_SIZE) {
      return false;
    }
    Object[] inputs = getInputs();
    for (int i = 0; i < outputs.length; i++) {
      float[] input = (float[]) inputs[i];
      float[] output = (float[]) outputs[i];
      if (!validateOutput(input, output)) {
        return false;
      }
    }
    return true;
  }

  private boolean validateOutput(float[] input, float[] output) {
    if (output.length != MODEL_OUTPUT_DIMENSIONS) {
      logger.info(
          "Output sizes do not match: Got "
              + output.length
              + "-  Expected: "
              + MODEL_OUTPUT_DIMENSIONS);
      return false;
    }
    for (int i = 0; i < MODEL_OUTPUT_DIMENSIONS; i++) {
      float got = output[i];
      float expected = input[i] * 3;
      if (Float.compare(got, expected) != 0) {
        logger.info("Values do not match; Got: " + got + "- Expected: " + expected);
        return false;
      }
    }
    logger.info("Printing first 10 elements: ");
    logger.info(Arrays.toString(Arrays.copyOfRange(output, 0, 10)));
    return true;
  }

  /** Generates sample input. */
  private static float generateInput(int x) {
    return 55.25f + 2 * x;
  }

  private static final int product(int... values) {
    int total = 1;
    for (int v : values) {
      total *= v;
    }
    return total;
  }

  private static float[] toFloatArray(ByteBuffer output) {
    FloatBuffer outputFloats = output.asFloatBuffer();
    float[] predictions = new float[outputFloats.remaining()];
    outputFloats.get(predictions);
    return predictions;
  }
}
