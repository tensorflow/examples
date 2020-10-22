/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.imagesegmentation;

import static com.google.common.truth.Truth.assertThat;

import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.util.Log;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;
import java.io.IOException;
import java.io.InputStream;
import java.util.Collections;
import java.util.HashSet;
import java.util.Set;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.tensorflow.lite.examples.imagesegmentation.tflite.ImageSegmentationModelExecutor;
import org.tensorflow.lite.examples.imagesegmentation.tflite.ModelExecutionResult;

/** Golden test for Image Segmentation Reference app. */
@RunWith(AndroidJUnit4.class)
public class ImageSegmentationModelExecutorTest {

  private static final String INPUT = "input_image.jpg";
  private static final String GOLDEN_OUTPUTS_TASK = "golden_output_task.png";
  private static final double GOLDEN_MASK_TOLERANCE = 1e-2;
  private static final String[] goldenLabelArray = {"background", "person", "horse"};

  @Test
  public void executeResultsShouldNotChange() throws IOException {
    Context context = InstrumentationRegistry.getInstrumentation().getContext();
    ImageSegmentationModelExecutor segmenter = new ImageSegmentationModelExecutor(context, false);
    Bitmap input = loadImage(INPUT);

    ModelExecutionResult result = segmenter.execute(input);

    // Verify the output mask bitmap.
    if (ImageSegmentationModelExecutor.TAG.equals("SegmentationTask")) {
      String goldenOutputFileName = GOLDEN_OUTPUTS_TASK;

      int[] goldenPixels = getPixels(loadImage(goldenOutputFileName));
      int[] resultPixels = getPixels(result.getBitmapMaskOnly());

      assertThat(resultPixels).hasLength(goldenPixels.length);
      int inconsistentPixels = 0;
      for (int i = 0; i < resultPixels.length; i++) {
        if (resultPixels[i] != goldenPixels[i]) {
          inconsistentPixels++;
        }
      }
      assertThat((double) inconsistentPixels / resultPixels.length)
          .isLessThan(GOLDEN_MASK_TOLERANCE);
    }

    // Verify labels.
    Set<String> resultLabels = new HashSet<>();
    for (String itemName : result.getItemsFound().keySet()) {
      resultLabels.add(itemName);
    }

    Set<String> goldenLabels = new HashSet<>();
    Collections.addAll(goldenLabels, goldenLabelArray);
    assertThat(resultLabels).isEqualTo(goldenLabels);
  }

  private static int[] getPixels(Bitmap bitmap) {
    int width = bitmap.getWidth();
    int height = bitmap.getHeight();
    int[] pixels = new int[width * height];
    bitmap.getPixels(pixels, 0, width, 0, 0, width, height);
    return pixels;
  }

  private static Bitmap loadImage(String fileName) {
    AssetManager assetManager =
        InstrumentationRegistry.getInstrumentation().getContext().getAssets();
    InputStream inputStream = null;
    try {
      inputStream = assetManager.open(fileName);
    } catch (IOException e) {
      Log.e("Test", "Cannot load image from assets");
    }
    return BitmapFactory.decodeStream(inputStream);
  }
}
