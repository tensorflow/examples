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

import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.robolectric.RobolectricTestRunner;

/** Tests of {@link org.tensorflow.lite.examples.bertqa.toeknization.BasicTokenizer} */
@RunWith(RobolectricTestRunner.class)
public final class BasicTokenizerTest {
  @Test
  public void cleanTextTest() throws Exception {
    String testExample = "This is an\rexample.\n";
    char testChar = 0;
    char unknownChar = 0xfffd;
    assertThat(BasicTokenizer.cleanText(testExample)).isEqualTo("This is an example. ");
    assertThat(BasicTokenizer.cleanText(testExample + testChar)).isEqualTo("This is an example. ");
    assertThat(BasicTokenizer.cleanText(testExample + unknownChar))
        .isEqualTo("This is an example. ");

    String nullString = null;
    Assert.assertThrows(NullPointerException.class, () -> BasicTokenizer.cleanText(nullString));
  }

  @Test
  public void whitespaceTokenizeTest() throws Exception {
    assertThat(BasicTokenizer.whitespaceTokenize("Hi , This is an example. "))
        .containsExactly("Hi", ",", "This", "is", "an", "example.")
        .inOrder();
    assertThat(BasicTokenizer.whitespaceTokenize("       ")).isEmpty();

    String nullString = null;
    Assert.assertThrows(
        NullPointerException.class, () -> BasicTokenizer.whitespaceTokenize(nullString));
  }

  @Test
  public void runSplitOnPuncTest() throws Exception {
    assertThat(BasicTokenizer.runSplitOnPunc("Hi,there."))
        .containsExactly("Hi", ",", "there", ".")
        .inOrder();
    assertThat(BasicTokenizer.runSplitOnPunc("I'm \"Spider-Man\"")) // Input: I'm "Spider-Man"
        .containsExactly("I", "\'", "m ", "\"", "Spider", "-", "Man", "\"")
        .inOrder();

    String nullString = null;
    Assert.assertThrows(
        NullPointerException.class, () -> BasicTokenizer.runSplitOnPunc(nullString));
  }

  @Test
  public void tokenizeWithLowerCaseTest() throws Exception {
    BasicTokenizer tokenizer = new BasicTokenizer(/*doLowerCase=*/ true);
    assertThat(tokenizer.tokenize("  Hi, This\tis an example.\n"))
        .containsExactly("hi", ",", "this", "is", "an", "example", ".")
        .inOrder();
    assertThat(tokenizer.tokenize("Hello,How are you?"))
        .containsExactly("hello", ",", "how", "are", "you", "?")
        .inOrder();

    String nullString = null;
    Assert.assertThrows(NullPointerException.class, () -> tokenizer.tokenize(nullString));
  }

  @Test
  public void tokenizeTest() throws Exception {
    BasicTokenizer tokenizer = new BasicTokenizer(/*doLowerCase=*/ false);
    assertThat(tokenizer.tokenize("  Hi, This\tis an example.\n"))
        .containsExactly("Hi", ",", "This", "is", "an", "example", ".")
        .inOrder();
    assertThat(tokenizer.tokenize("Hello,How are you?"))
        .containsExactly("Hello", ",", "How", "are", "you", "?")
        .inOrder();

    String nullString = null;
    Assert.assertThrows(NullPointerException.class, () -> tokenizer.tokenize(nullString));
  }
}
