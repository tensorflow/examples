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
import com.google.android.gms.tasks.Task;
import com.google.android.gms.tasks.Tasks;
import java.util.concurrent.Executor;
import org.tensorflow.lite.examples.accelerationservice.logger.Logger;

/** Loads tflite models stored as assets. */
public final class AssetModelFactory {

  private final Context context;
  private final Executor executor;
  private final Logger logger;

  public AssetModelFactory(Context context, Executor executor, Logger logger) {
    this.context = context;
    this.executor = executor;
    this.logger = logger;
  }

  /** Asset models. */
  public enum ModelType {
    PLAIN_ADDITION,
    MOBILENET_V1,
  }

  /** Loads model from assets. */
  public Task<AssetModel> load(ModelType type) {
    switch (type) {
      case PLAIN_ADDITION:
        return Tasks.forResult(null)
            .onSuccessTask(executor, unused -> Tasks.forResult(new PlainAddition(context, logger)));
      case MOBILENET_V1:
        return Tasks.forResult(null)
            .onSuccessTask(executor, unused -> Tasks.forResult(new MobileNetV1(context, logger)));
      default:
        throw new IllegalArgumentException("Invalid ModelType: " + type);
    }
  }
}
