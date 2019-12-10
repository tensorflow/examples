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
import java.nio.MappedByteBuffer;
import org.tensorflow.lite.Interpreter;

/**
 * Superclass for TFLite model wrappers that handles model resource management.
 */
final class LiteModelWrapper implements Closeable {
  private final Interpreter interpreter;

  private LiteModelWrapper(ByteBuffer model) {
    interpreter = new Interpreter(model);
  }

  /**
   * Create a model wrapper and an interpreter instance.
   * @param model raw model data, with no additional restrictions.
   */
  LiteModelWrapper(byte[] model) {
    this(convertToDirectBuffer(model));
  }

  /**
   * Create a model wrapper and an interpreter instance.
   * @param model raw model data, mmap-ed from a file
   */
  LiteModelWrapper(MappedByteBuffer model) {
    this((ByteBuffer) model);
  }

  Interpreter getInterpreter() {
    return interpreter;
  }

  @Override
  public void close() {
    interpreter.close();
  }

  private static ByteBuffer convertToDirectBuffer(byte[] data) {
    ByteBuffer result = ByteBuffer.allocateDirect(data.length);
    result.order(ByteOrder.nativeOrder());
    result.put(data);
    result.rewind();
    return result;
  }
}
