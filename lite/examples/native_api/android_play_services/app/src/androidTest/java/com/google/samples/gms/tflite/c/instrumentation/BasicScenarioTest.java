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

package com.google.samples.gms.tflite.c.instrumentation;

import static com.google.common.truth.Truth.assertThat;

import android.content.Context;
import android.util.Log;
import androidx.test.core.app.ApplicationProvider;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import com.google.android.gms.tasks.Tasks;
import com.google.android.gms.tflite.java.TfLiteNative;
import com.google.samples.gms.tflite.c.TfLiteJni;
import java.util.concurrent.ExecutionException;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Instrumentation tests for the TFLite Native API. */
@RunWith(AndroidJUnit4.class)
public class BasicScenarioTest {
  private static final String TAG = "BasicScenarioTest";

  @Test
  public void basicScenario() throws ExecutionException, InterruptedException {
    Context context = ApplicationProvider.getApplicationContext();
    Tasks.await(TfLiteNative.initialize(context));
    TfLiteJni jni = new TfLiteJni(message -> Log.e(TAG, message));

    jni.loadModel(context.getAssets(), "add.tflite");
    float[] output = jni.runInference(new float[] {1.f, 3.f});
    jni.destroy();

    assertThat(output).isEqualTo(new float[] {3.f, 9.f});
  }
}
