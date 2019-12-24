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

import java.io.Closeable;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Map;
import java.util.TreeMap;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;

/**
 * A wrapper for TFLite model that calculates the gradients of trainable layers.
 */
class LiteTrainHeadModel implements Closeable {
  private static final int FLOAT_BYTES = 4;

  private LiteModelWrapper modelWrapper;

  LiteTrainHeadModel(LiteModelWrapper modelWrapper) {
    this.modelWrapper = modelWrapper;
  }

  /**
   * Performs single training pass (forward + backward).
   *
   * @param bottleneckBatch image bottlenecks.
   * @param classBatch ground truth labels for images.
   * @param modelParameters current model trainable parameter values.
   * @param modelGradients where to store model trainable parameter gradients.
   * @return loss
   */
  float calculateGradients(
      ByteBuffer bottleneckBatch,
      ByteBuffer classBatch,
      ByteBuffer[] modelParameters,
      ByteBuffer[] modelGradients) {
    if (modelParameters.length != modelGradients.length) {
      throw new IllegalArgumentException(String.format(
          "Parameter array size (%d) is different from gradient array size (%d)",
          modelParameters.length,
          modelGradients.length));
    }
    if (modelWrapper.getInterpreter().getOutputTensorCount() != modelParameters.length + 1) {
      throw new IllegalArgumentException(String.format(
          "Model expected %d parameter tensors, but got %d",
          modelWrapper.getInterpreter().getInputTensorCount() - 1,
          modelParameters.length));
    }

    ByteBuffer lossBuffer = ByteBuffer.allocateDirect(FLOAT_BYTES);
    lossBuffer.order(ByteOrder.nativeOrder());

    Map<Integer, Object> outputs = new TreeMap<>();
    outputs.put(0, lossBuffer);
    for (int outputIndex = 1;
        outputIndex < modelWrapper.getInterpreter().getOutputTensorCount();
        outputIndex++) {
      outputs.put(outputIndex, modelGradients[outputIndex - 1]);
    }

    Object[] inputs = new Object[modelParameters.length + 2];
    inputs[0] = bottleneckBatch;
    inputs[1] = classBatch;
    System.arraycopy(modelParameters, 0, inputs, 2, modelParameters.length);

    modelWrapper.getInterpreter().runForMultipleInputsOutputs(inputs, outputs);

    bottleneckBatch.rewind();
    classBatch.rewind();

    for (ByteBuffer buffer : modelParameters) {
      buffer.rewind();
    }
    for (ByteBuffer buffer : modelGradients) {
      buffer.rewind();
    }

    lossBuffer.rewind();
    return lossBuffer.getFloat();
  }

  int getBatchSize() {
    return modelWrapper.getInterpreter().getInputTensor(0).shape()[0];
  }

  int[] getParameterSizes() {
    int[] parameterSizes = new int[modelWrapper.getInterpreter().getInputTensorCount() - 2];
    for (int inputIndex = 2;
        inputIndex < modelWrapper.getInterpreter().getInputTensorCount();
        inputIndex++) {
      parameterSizes[inputIndex - 2] =
          modelWrapper.getInterpreter().getInputTensor(inputIndex).numElements();
    }
    return parameterSizes;
  }

  int[][] getParameterShapes() {
    Interpreter interpreter = modelWrapper.getInterpreter();

    int[][] parameterShapes = new int[interpreter.getInputTensorCount() - 2][];
    for (int inputIndex = 2; inputIndex < interpreter.getInputTensorCount(); inputIndex++) {
      Tensor inputTensor = interpreter.getInputTensor(inputIndex);

      parameterShapes[inputIndex - 2] = new int[inputTensor.numDimensions()];
      System.arraycopy(
          inputTensor.shape(), 0,
          parameterShapes[inputIndex - 2], 0,
          inputTensor.numDimensions());
    }

    return parameterShapes;
  }

  @Override
  public void close() {
    modelWrapper.close();
  }
}
