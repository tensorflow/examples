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

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;
import java.io.BufferedReader;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.Charset;
import java.util.Arrays;
import java.util.Iterator;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.tensorflow.lite.examples.transfer.api.TransferLearningModel.Prediction;

/** End-to-end model correctness test. */
@RunWith(AndroidJUnit4.class)
public class ModelCorrectnessTest {
  private static final int LOWER_BYTE_MASK = 0xFF;

  private static final int NUM_EPOCHS = 20;
  private static final float TARGET_ACCURACY = 0.70f;

  private static class Sample {
    String imagePath;
    String className;

    Sample(String imagePath, String className) {
      this.imagePath = imagePath;
      this.className = className;
    }
  }

  @Test
  public void shouldLearnToClassifyImages() throws IOException {
    Map<String, byte[]> zipFiles =
        ZipUtils.readAllZipFiles(
            InstrumentationRegistry.getInstrumentation().getContext(), "test_data.zip");

    TransferLearningModel model =
        new TransferLearningModel(
            new AssetModelLoader(
                InstrumentationRegistry.getInstrumentation().getContext(), "model"),
            Arrays.asList("daisy", "dandelion", "roses", "sunflowers", "tulips"));

    System.out.println("Going to add the samples.");

    for (Sample sample : readSampleList(zipFiles.get("train.txt"))) {
      try {
        model.addSample(jpgBytesToRgb(zipFiles.get(sample.imagePath)), sample.className).get();
      } catch (InterruptedException e) {
        return;
      } catch (ExecutionException e) {
        throw new RuntimeException("Could not add training sample", e.getCause());
      }
    }

    System.out.println("Finished adding the samples.");

    try {
      model
          .train(
              NUM_EPOCHS,
              (epoch, loss) -> {
                System.out.printf("Epoch %d: loss = %.5f\n", epoch, loss);
              })
          .get();
    } catch (ExecutionException e) {
      throw new RuntimeException(e.getCause());
    } catch (InterruptedException e) {
      // Exit peacefully.
    }

    int correct = 0;
    int total = 0;
    for (Sample sample : readSampleList(zipFiles.get("val.txt"))) {
      Prediction[] predictions = model.predict(jpgBytesToRgb(zipFiles.get(sample.imagePath)));
      if (predictions[0].getClassName().equals(sample.className)) {
        correct++;
      }
      total++;
    }

    float accuracy = correct / (float) total;
    System.out.printf("Accuracy is %.5f\n", accuracy);
    assertTrue(
        String.format("Accuracy is %.5f, expected at least %.5f", accuracy, TARGET_ACCURACY),
        accuracy >= TARGET_ACCURACY);
  }

  private Iterable<Sample> readSampleList(byte[] sampleListBytes) {
    BufferedReader linesReader = new BufferedReader(new InputStreamReader(
        new ByteArrayInputStream(sampleListBytes), Charset.defaultCharset()));

    return () -> new Iterator<Sample>() {
      private boolean hasBufferedLine = false;
      private String nextLine = null;

      @Override
      public boolean hasNext() {
        maybeReadToBuffer();
        return nextLine != null;
      }

      @Override
      public Sample next() {
        maybeReadToBuffer();
        hasBufferedLine = false;

        String[] parts = nextLine.split(",", -1);
        return new Sample(parts[0], parts[1]);
      }

      private void maybeReadToBuffer() {
        if (!hasBufferedLine) {
          try {
            nextLine = linesReader.readLine();
            hasBufferedLine = true;
          } catch (IOException e) {
            throw new RuntimeException(e);
          }
        }
      }
    };
  }

  private static float[] jpgBytesToRgb(byte[] jpgBytes) throws IOException {
    ByteArrayInputStream inputStream = new ByteArrayInputStream(jpgBytes);
    Bitmap image = BitmapFactory.decodeStream(inputStream);

    float[] result = new float[image.getWidth() * image.getHeight() * 3];
    int nextIdx = 0;

    for (int y = 0; y < image.getHeight(); y++) {
      for (int x = 0; x < image.getWidth(); x++) {
        int rgb = image.getPixel(x, y);

        float r = ((rgb >> 16) & LOWER_BYTE_MASK) * (1 / 255.f);
        float g = ((rgb >> 8) & LOWER_BYTE_MASK) * (1 / 255.f);
        float b = (rgb & LOWER_BYTE_MASK) * (1 / 255.f);

        result[nextIdx++] = r;
        result[nextIdx++] = g;
        result[nextIdx++] = b;
      }
    }

    return result;
  }
}
