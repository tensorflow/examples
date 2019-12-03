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

package org.tensorflow.lite.examples.transfer;

import android.util.Log;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;

/**
 * A simple class for measuring execution time in various contexts.
 */
class LoggingBenchmark {
  private static final boolean ENABLED = false;

  private final String tag;

  private final Map<String, Long> totalImageTime = new HashMap<>();
  private final Map<String, Map<String, Long>> stageTime = new HashMap<>();

  private final Map<String, Map<String, Long>> stageStartTime = new HashMap<>();

  LoggingBenchmark(String tag) {
    this.tag = tag;
  }

  void startStage(String imageId, String stageName) {
    if (!ENABLED) {
      return;
    }

    Map<String, Long> stageStartTimeForImage;
    if (!stageStartTime.containsKey(imageId)) {
      stageStartTimeForImage = new HashMap<>();
      stageStartTime.put(imageId, stageStartTimeForImage);
    } else {
      stageStartTimeForImage = stageStartTime.get(imageId);
    }

    long timeNs = System.nanoTime();
    stageStartTimeForImage.put(stageName, timeNs);
  }

  void endStage(String imageId, String stageName) {
    if (!ENABLED) {
      return;
    }

    long endTime = System.nanoTime();
    long startTime = stageStartTime.get(imageId).get(stageName);
    long duration = endTime - startTime;

    if (!stageTime.containsKey(imageId)) {
      stageTime.put(imageId, new HashMap<>());
    }
    stageTime.get(imageId).put(stageName, duration);

    if (!totalImageTime.containsKey(imageId)) {
      totalImageTime.put(imageId, 0L);
    }
    totalImageTime.put(imageId, totalImageTime.get(imageId) + duration);
  }

  void finish(String imageId) {
    if (!ENABLED) {
      return;
    }

    StringBuilder msg = new StringBuilder();
    for (Map.Entry<String, Long> entry : stageTime.get(imageId).entrySet()) {
      msg.append(String.format(Locale.getDefault(),
          "%s: %.2fms | ", entry.getKey(), entry.getValue() / 1.0e6));
    }

    msg.append(String.format(Locale.getDefault(),
        "TOTAL: %.2fms", totalImageTime.get(imageId) / 1.0e6));
    Log.d(tag, msg.toString());
  }
}
