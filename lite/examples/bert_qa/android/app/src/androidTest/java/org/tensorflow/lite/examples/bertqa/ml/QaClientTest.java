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
package org.tensorflow.lite.examples.bertqa.ml;

import static com.google.common.truth.Truth.assertThat;

import androidx.test.core.app.ApplicationProvider;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import java.util.ArrayList;
import java.util.List;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Tests of {@link org.tensorflow.lite.examples.bertqa.ml.QaClient} */
@RunWith(AndroidJUnit4.class)
public final class QaClientTest {
  private QaClient client;

  @Before
  public void setUp() {
    client = new QaClient(ApplicationProvider.getApplicationContext());
    client.loadModel();
  }

  @Test
  public void testGoldSet() {
    final String content =
        "Nikola Tesla (Serbian Cyrillic: \u041d\u0438\u043a\u043e\u043b\u0430"
            + " \u0422\u0435\u0441\u043b\u0430; 10 July 1856 \u2013 7 January 1943) was a Serbian"
            + " American inventor, electrical engineer, mechanical engineer, physicist, and"
            + " futurist best known for his contributions to the design of the modern alternating"
            + " current (AC) electricity supply system.";

    List<QaAnswer> predict0 = client.predict("What is Tesla's home country?", content);
    assertThat(getTexts(predict0)).contains("Serbian");
    List<QaAnswer> predict1 = client.predict("What was Nikola Tesla's ethnicity?", content);
    assertThat(getTexts(predict1)).contains("Serbian");
    List<QaAnswer> predict2 = client.predict("What does AC stand for?", content);
    assertThat(getTexts(predict2)).contains("alternating current");
    List<QaAnswer> predict3 = client.predict("When was Tesla born?", content);
    assertThat(getTexts(predict3)).contains("10 July 1856");
  }

  private static List<String> getTexts(List<QaAnswer> answers) {
    List<String> texts = new ArrayList<>();
    for (QaAnswer ans : answers) {
      texts.add(ans.text);
    }
    return texts;
  }
}
