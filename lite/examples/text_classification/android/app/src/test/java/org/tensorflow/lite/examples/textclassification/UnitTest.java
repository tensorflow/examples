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

package org.tensorflow.lite.examples.textclassification;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import androidx.test.core.app.ApplicationProvider;
import java.util.Arrays;
import java.util.List;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.robolectric.RobolectricTestRunner;
import org.tensorflow.lite.examples.textclassification.TextClassificationClient.Result;

/** Tests of {@link org.tensorflow.lite.examples.textclassification.TextClassificationClient} */
@RunWith(RobolectricTestRunner.class)
public final class UnitTest {
  private TextClassificationClient client;
  private static final String MODEL_PATH = "text_classification.tflite";

  @Before
  public void setUp() {
    client = new TextClassificationClient(ApplicationProvider.getApplicationContext(), MODEL_PATH);
    client.load();
  }

  @Test
  public void loadModelTest() {
    assertNotNull(client.getTflite());
  }

  @Test
  public void loadDictinaryTest() {
    assertEquals(0, (int) client.getDic().get("<PAD>"));
    assertEquals(1, (int) client.getDic().get("<START>"));
    assertEquals(2, (int) client.getDic().get("<UNKNOWN>"));
    assertEquals(3, (int) client.getDic().get("the"));
  }

  @Test
  public void loadLabelsTest() {
    List<String> labels = client.getLabels();
    assertEquals("Negative", labels.get(0));
    assertEquals("Positive", labels.get(1));
  }

  @Test
  public void inputPreprocessingTest() {
    int[][] clientOutput = client.tokenizeInputText("hello,world!");
    int[][] expectOutput = new int[1][256];
    Arrays.fill(expectOutput[0], 0, 255, 0);
    expectOutput[0][0] = 1; // Index for <START>
    expectOutput[0][1] = 4845; // Index for "hello".
    expectOutput[0][2] = 181; // Index for "world".
    assertArrayEquals(expectOutput, clientOutput);
  }

  @Test
  public void predictTest() {
    Result positiveText =
        client
            .classify("This is an interesting film. My family and I all liked it very much.")
            .get(0);
    assertEquals("Positive", positiveText.getTitle());
    assertTrue(positiveText.getConfidence() > 0.55);
    Result negativeText =
        client.classify("This film cannot be worse. It is way too boring.").get(0);
    assertEquals("Negative", negativeText.getTitle());
    assertTrue(negativeText.getConfidence() > 0.6);
  }
}
