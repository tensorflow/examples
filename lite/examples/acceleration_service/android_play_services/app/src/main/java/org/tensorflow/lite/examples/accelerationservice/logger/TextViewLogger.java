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

package org.tensorflow.lite.examples.accelerationservice.logger;

import android.util.Log;
import android.widget.TextView;
import androidx.annotation.Nullable;
import androidx.annotation.UiThread;
import androidx.appcompat.app.AppCompatActivity;
import java.io.PrintWriter;
import java.io.StringWriter;

/** Logs messages to the {@code TextView} component. */
public class TextViewLogger implements Logger {

  private static final String TAG = "Logger";

  private final TextView output;
  private final AppCompatActivity activity;

  public TextViewLogger(AppCompatActivity activity, TextView output) {
    this.activity = activity;
    this.output = output;
  }

  @Override
  public void error(String message, Exception e) {
    activity.runOnUiThread((Runnable) () -> logEvent(message, e));
  }

  @Override
  public void info(String message) {
    activity.runOnUiThread((Runnable) () -> logEvent(message, null));
  }

  @Override
  @UiThread
  public void clear() {
    output.setText("");
  }

  @UiThread
  private void logEvent(String message, @Nullable Throwable throwable) {
    Log.e(TAG, message, throwable);
    output.append("• ");
    output.append(String.valueOf(message)); // valueOf converts null to "null"
    output.append("\n");
    if (throwable != null) {
      StringWriter sw = new StringWriter();
      throwable.printStackTrace(new PrintWriter(sw));
      output.append(sw.toString());
    }
  }
}
