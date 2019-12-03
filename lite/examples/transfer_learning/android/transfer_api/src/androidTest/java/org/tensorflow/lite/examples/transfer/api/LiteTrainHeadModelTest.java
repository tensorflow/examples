/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package org.tensorflow.lite.examples.transfer.api;

import static org.junit.Assert.assertTrue;

import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Random;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Gradient calculation test for {@link LiteTrainHeadModel}. */
@RunWith(AndroidJUnit4.class)
public class LiteTrainHeadModelTest {
  private interface Supplier<T> {
    T get();
  }

  private static final int FLOAT_BYTES = 4;

  private static final int BATCH_SIZE = 20;
  private static final int BOTTLENECK_SIZE = 7 * 7 * 1280;
  private static final int NUM_CLASSES = 5;

  // Only first N elements of every parameter will be checked to save time.
  private static final int MAX_ELEMENTS_TO_CHECK = 30;

  private static final float DELTA_PARAM = 3e-3f;
  private static final float EPS = 1e-2f;

  private static final Random random = new Random(32);

  @Test
  public void shouldCalculateGradientsCorrectly() throws IOException {
    LiteTrainHeadModel model =
        new LiteTrainHeadModel(
            new AssetModelLoader(InstrumentationRegistry.getInstrumentation().getContext(), "model")
                .loadTrainModel());

    ByteBuffer bottlenecks = generateRandomByteBuffer(
        BATCH_SIZE * BOTTLENECK_SIZE, random::nextFloat);
    ByteBuffer classes = generateRandomByteBuffer(BATCH_SIZE * NUM_CLASSES, () -> 0.f);

    int[][] parameterShapes = model.getParameterShapes();
    ByteBuffer[] parameters = new ByteBuffer[parameterShapes.length];
    for (int parameterIdx = 0; parameterIdx < parameterShapes.length; parameterIdx++) {
      parameters[parameterIdx] = generateRandomByteBuffer(parameterShapes[parameterIdx]);
    }

    // Fill with one-hot.
    for (int sampleIdx = 0; sampleIdx < BATCH_SIZE; sampleIdx++) {
      int sampleClass = random.nextInt(NUM_CLASSES);
      classes.putFloat((sampleIdx * NUM_CLASSES + sampleClass) * FLOAT_BYTES, 1);
    }

    ByteBuffer[] gradients = new ByteBuffer[parameterShapes.length];
    for (int parameterIdx = 0; parameterIdx < parameterShapes.length; parameterIdx++) {
      gradients[parameterIdx] = generateRandomByteBuffer(parameterShapes[parameterIdx]);
    }

    float loss = model.calculateGradients(bottlenecks, classes, parameters, gradients);

    for (int parameterIdx = 0; parameterIdx < parameters.length; parameterIdx++) {
      ByteBuffer parameter = parameters[parameterIdx];
      ByteBuffer analyticalGrads = gradients[parameterIdx];
      int numElementsToCheck =
          Math.min(product(parameterShapes[parameterIdx]), MAX_ELEMENTS_TO_CHECK);

      for (int elementIdx = 0; elementIdx < numElementsToCheck; elementIdx++) {
        float analyticalGrad = analyticalGrads.getFloat(elementIdx * FLOAT_BYTES);

        float originalParam = parameter.getFloat(elementIdx * FLOAT_BYTES);
        parameter.putFloat(elementIdx * FLOAT_BYTES, originalParam + DELTA_PARAM);
        float newLoss = model.calculateGradients(bottlenecks, classes, parameters, gradients);

        float numericalGrad = (newLoss - loss) / DELTA_PARAM;
        assertTrue(
            String.format("Numerical gradient %.5f is different from analytical %.5f "
                + "(loss = %.5f -> %.5f)",
                numericalGrad, analyticalGrad, loss, newLoss),
            Math.abs(numericalGrad - analyticalGrad) < EPS);

        parameter.putFloat(elementIdx * FLOAT_BYTES, originalParam);
      }
    }
  }

  private static ByteBuffer generateRandomByteBuffer(int[] tensorShape) {
    float stdDev;
    if (tensorShape.length >= 2) {
      stdDev = (float) Math.sqrt(2. / (tensorShape[0] + tensorShape[1]));
    } else {
      stdDev = 0;
    }

    return generateRandomByteBuffer(
        product(tensorShape), () -> (float) random.nextGaussian() * stdDev);
  }

  private static ByteBuffer generateRandomByteBuffer(int numElements, Supplier<Float> initializer) {
    ByteBuffer result = ByteBuffer.allocateDirect(numElements * FLOAT_BYTES);
    result.order(ByteOrder.nativeOrder());

    for (int idx = 0; idx < numElements; idx++) {
      result.putFloat(initializer.get());
    }
    result.rewind();

    return result;
  }

  private static int product(int[] array) {
    int result = 1;
    for (int element : array) {
      result *= element;
    }
    return result;
  }
}
