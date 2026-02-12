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

import android.app.Activity;
import android.os.Bundle;
import android.util.Log;
import android.widget.TextView;
import androidx.annotation.Nullable;
import com.google.android.gms.tasks.Task;
import com.google.android.gms.tasks.Tasks;
import com.google.android.gms.tflite.client.TfLiteInitializationOptions;
import com.google.android.gms.tflite.gpu.support.TfLiteGpu;
import com.google.android.gms.tflite.java.TfLiteNative;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Date;

/** Sample activity to test the TFLite C API. */
public class MainActivity extends Activity {

  private static final String TAG = "MainActivity";

  private volatile boolean isGpuAvailable = false;

  private TextView logView;

  @Override
  protected void onCreate(@Nullable Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);

    logView = findViewById(R.id.log_text);

    findViewById(android.R.id.button1).setOnClickListener(v -> runScenario());
  }

  @Override
  protected void onStart() {
    super.onStart();
    runScenario();
  }

  private void runScenario() {
    String currentTime = SimpleDateFormat.getTimeInstance(DateFormat.SHORT).format(new Date());
    logView.setText(String.format("Scenario started at %s\n", currentTime));
    isGpuAvailable = false;

    logEvent("Checking GPU acceleration availability...");

    Task<Void> tfLiteHandleTask =
        TfLiteGpu.isGpuDelegateAvailable(this)
            .onSuccessTask(
                gpuAvailable -> {
                  isGpuAvailable = gpuAvailable;
                  logEvent("GPU acceleration is " + (isGpuAvailable ? "available" : "unavailable"));
                  TfLiteInitializationOptions options =
                      TfLiteInitializationOptions.builder()
                          .setEnableGpuDelegateSupport(isGpuAvailable)
                          .build();
                  return TfLiteNative.initialize(this, options);
                });

    tfLiteHandleTask
        .onSuccessTask(
            unused -> {
              logEvent("Running inference on " + (isGpuAvailable ? "GPU" : "CPU"));
              TfLiteJni jni = new TfLiteJni(this::logEvent);
              if (isGpuAvailable) {
                jni.initGpuAcceleration();
              }
              logEvent("TfLiteJni created");
              jni.loadModel(getAssets(), "add.tflite");
              logEvent("Model loaded");
              float[] output = jni.runInference(new float[] {1.f, 3.f});
              logEvent(
                  "Ran inference, expected: [3.0, 9.0], got output: " + Arrays.toString(output));
              jni.destroy();
              logEvent("TfLiteJni destroyed");
              return Tasks.forResult(output);
            })
        .addOnSuccessListener(unused -> logEvent("Scenario successful!"))
        .addOnFailureListener(e -> logEvent("Scenario failed", e));
  }

  private void logEvent(String message) {
    logEvent(message, null);
  }

  private void logEvent(String message, @Nullable Throwable throwable) {
    Log.e(TAG, message, throwable);
    logView.append("• ");
    logView.append(String.valueOf(message));
    logView.append("\n");
    if (throwable != null) {
      logView.append(throwable.getClass().getCanonicalName() + ": " + throwable.getMessage());
      logView.append("\n");
      logView.append(Arrays.toString(throwable.getStackTrace()));
      logView.append("\n");
    }
  }
}
