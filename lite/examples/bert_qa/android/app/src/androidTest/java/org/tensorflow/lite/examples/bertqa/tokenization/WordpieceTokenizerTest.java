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
package org.tensorflow.lite.examples.bertqa.tokenization;

import static com.google.common.truth.Truth.assertThat;

import androidx.test.core.app.ApplicationProvider;
import java.util.Map;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.robolectric.RobolectricTestRunner;
import org.tensorflow.lite.examples.bertqa.ml.LoadDatasetClient;

/** Tests of {@link org.tensorflow.lite.examples.bertqa.tokenization.WordpieceTokenizer} */
@RunWith(RobolectricTestRunner.class)
public final class WordpieceTokenizerTest {
  private Map<String, Integer> dic;

  @Before
  public void setUp() {
    LoadDatasetClient client = new LoadDatasetClient(ApplicationProvider.getApplicationContext());
    dic = client.loadDictionary();
  }

  @Test
  public void tokenizeTest() throws Exception {
    WordpieceTokenizer tokenizer = new WordpieceTokenizer(dic);
    assertThat(tokenizer.tokenize("meaningfully")).containsExactly("meaningful", "##ly").inOrder();
    assertThat(tokenizer.tokenize("teacher")).containsExactly("teacher").inOrder();

    String nullString = null;
    Assert.assertThrows(NullPointerException.class, () -> tokenizer.tokenize(nullString));
  }
}
