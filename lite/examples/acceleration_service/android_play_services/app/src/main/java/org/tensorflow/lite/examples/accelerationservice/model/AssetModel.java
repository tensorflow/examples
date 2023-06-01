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

import com.google.android.gms.tflite.acceleration.Model;
import java.nio.ByteBuffer;
import java.util.Map;

/** Provides methods for interacting with tflite models stored as assets. */
public interface AssetModel {

  /** Returns {@link Model} instance. */
  Model getModel();

  /** Returns sample input batch size. */
  int getBatchSize();

  /** Returns sample input. */
  Object[] getInputs();

  /** Returns sample output allocation. */
  Map<Integer, Object> allocateOutputs();

  /** Checks if the benchmark model output is valid. */
  boolean validateBenchmarkOutputs(ByteBuffer[] outputs);

  /** Checks if the {@link InterpreterApi} model output is valid. */
  boolean validateInterpreterOutputs(Object[] outputs);
}
