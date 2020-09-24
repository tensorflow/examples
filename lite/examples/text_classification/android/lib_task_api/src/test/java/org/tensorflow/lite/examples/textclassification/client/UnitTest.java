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

package org.tensorflow.lite.examples.textclassification.client;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import androidx.test.core.app.ApplicationProvider;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.robolectric.RobolectricTestRunner;

/** Tests of {@link TextClassificationClient} */
@RunWith(RobolectricTestRunner.class)
public final class UnitTest {
  private TextClassificationClient client;

  @Before
  public void setUp() {
    client = new TextClassificationClient(ApplicationProvider.getApplicationContext());
    client.load();
  }

  @Test
  public void loadModelTest() {
    assertNotNull(client.classifier);
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
