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
import org.junit.Test;
import org.junit.runner.RunWith;

/** Smoke tests for {@link LiteBottleneckModel}. */
@RunWith(AndroidJUnit4.class)
public class LiteBottleneckModelTest {
  private static final int FLOAT_BYTES = 4;

  private static final int NUM_BOTTLENECK_FEATURES = 7 * 7 * 1280;
  private static final int IMAGE_SIZE = 224;
  private static final int NUM_IMAGE_CHANNELS = 3;
  private static final float IMAGE_FILL = 0.3f;
  private static final float EPS = 1e-8f;

  @Test
  public void shouldGenerateSaneBottlenecks() throws IOException {
    LiteBottleneckModel model =
        new LiteBottleneckModel(
            new AssetModelLoader(InstrumentationRegistry.getInstrumentation().getContext(), "model")
                .loadBaseModel());

    ByteBuffer image =
        ByteBuffer.allocateDirect(IMAGE_SIZE * IMAGE_SIZE * NUM_IMAGE_CHANNELS * FLOAT_BYTES);

    for (int idx = 0; idx < IMAGE_SIZE * IMAGE_SIZE * NUM_IMAGE_CHANNELS; idx++) {
      image.putFloat(IMAGE_FILL);
    }
    image.rewind();

    ByteBuffer bottleneck = model.generateBottleneck(image, null);
    int nonZeroCount = 0;
    for (int idx = 0; idx < NUM_BOTTLENECK_FEATURES; idx++) {
      float feature = bottleneck.getFloat();
      if (Math.abs(feature) > EPS) {
        nonZeroCount++;
      }
    }
    assertTrue(nonZeroCount > 0);
  }
}
