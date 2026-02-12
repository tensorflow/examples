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

package org.tensorflow.lite.examples.accelerationservice.validator;

import com.google.android.gms.tflite.acceleration.BenchmarkResult;
import com.google.android.gms.tflite.acceleration.CustomValidationConfig.AccuracyValidator;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import org.tensorflow.lite.examples.accelerationservice.logger.Logger;

/**
 * Accuracy validator that computes mean squared error of benchmark output and golden output. If the
 * MSE is above the defined threshold, validation run will be considered unsuccessful.
 */
public class MeanSquaredErrorValidator implements AccuracyValidator {

  private final double threshold;
  private final Logger logger;

  public MeanSquaredErrorValidator(Logger logger, double threshold) {
    this.logger = logger;
    this.threshold = threshold;
  }

  /**
   * Validates if {@code benchmarkResult} and {@code goldenOutputs} match. When {@link
   * CustomValidationConfig.Builder#setGoldenOutputs} is not invoked, {@code goldenOutputs} is
   * provided by the Acceleration SDK as the output of running golden config.
   */
  @Override
  public boolean validate(BenchmarkResult benchmarkResult, ByteBuffer[] goldenOutputs) {
    if (benchmarkResult.actualOutput().size() != goldenOutputs.length) {
      logger.info(
          "Accuracy validator: benchmark result and golden output"
              + " dimensions do not match."
              + " Benchmark result length: "
              + benchmarkResult.actualOutput().size()
              + " Golden output length: "
              + goldenOutputs.length);
      return false;
    }
    // Compare benchmark result buffer tensors with golden output tensors
    for (int i = 0; i < benchmarkResult.actualOutput().size(); i++) {
      FloatBuffer goldenOutputBuffer = goldenOutputs[i].asFloatBuffer();
      FloatBuffer benchmarkOutputBuffer =
          benchmarkResult.actualOutput().get(i).getValue().asFloatBuffer();
      boolean ok = checkMeanSquaredError(goldenOutputBuffer, benchmarkOutputBuffer);
      if (!ok) {
        logger.info(
            "Accuracy validator: benchmark result and golden output buffers do not match at"
                + " position: "
                + i);
        return false;
      }
    }
    logger.info("Accuracy validator: benchmark result and golden output match.");
    return true;
  }

  private boolean checkMeanSquaredError(FloatBuffer expected, FloatBuffer actual) {
    int length = expected.remaining();
    if (length != actual.remaining()) {
      logger.info(
          "Accuracy validator: benchmark result and golden output vector lengths do not"
              + " match. Got: "
              + actual.remaining()
              + " - Expected: "
              + expected.remaining()
              + ".");
      return false;
    }
    double sum = 0;
    for (int i = 0; i < length; i++) {
      sum += Math.pow(expected.get(i) - (double) actual.get(i), 2);
    }
    double mse = length == 0 ? 0 : sum / length;
    logger.info("Mean Squared Error: " + mse);
    return mse < threshold;
  }
}
