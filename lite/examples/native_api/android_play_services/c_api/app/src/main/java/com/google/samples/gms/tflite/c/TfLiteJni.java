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

package com.google.samples.gms.tflite.c;

import android.content.res.AssetManager;

/** JNI bridge to forward the calls to the native code, where we can invoke the TFLite C API. */
public class TfLiteJni {

  private final LoggingCallback loggingCallback;

  /**
   * This interface gets called when the JNI wants to print a message (used for debugging purposes).
   */
  public interface LoggingCallback {
    void printLogMessage(String message);
  }

  static {
    System.loadLibrary("tflite-jni");
  }

  public TfLiteJni(LoggingCallback loggingCallback) {
    this.loggingCallback = loggingCallback;
  }

  private void sendLogMessage(String message) {
    if (loggingCallback != null) {
      loggingCallback.printLogMessage(message);
    }
  }

  /** Creates GPU delegate that will be used for the inference. */
  public native void initGpuAcceleration();

  /**
   * Loads the model and creates the Interpreter. GPU delegate is applied if {@link
   * TfLiteJni#initGpuAcceleration} was previously called.
   */
  public native void loadModel(AssetManager assetManager, String assetName);

  /** Runs the inference using the Interpreter created by {@link TfLiteJni#loadModel}. */
  public native float[] runInference(float[] input);

  /** Unloads the assets and clears all the Interpreter's resources. */
  public native void destroy();
}
