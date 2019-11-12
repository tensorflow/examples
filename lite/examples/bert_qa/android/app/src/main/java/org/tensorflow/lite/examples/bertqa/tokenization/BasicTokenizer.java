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

import com.google.common.base.Ascii;
import com.google.common.collect.Iterables;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/** Basic tokenization (punctuation splitting, lower casing, etc.) */
public final class BasicTokenizer {
  private final boolean doLowerCase;

  public BasicTokenizer(boolean doLowerCase) {
    this.doLowerCase = doLowerCase;
  }

  public List<String> tokenize(String text) {
    String cleanedText = cleanText(text);

    List<String> origTokens = whitespaceTokenize(cleanedText);

    StringBuilder stringBuilder = new StringBuilder();
    for (String token : origTokens) {
      if (doLowerCase) {
        token = Ascii.toLowerCase(token);
      }
      List<String> list = runSplitOnPunc(token);
      for (String subToken : list) {
        stringBuilder.append(subToken).append(" ");
      }
    }
    return whitespaceTokenize(stringBuilder.toString());
  }

  /* Performs invalid character removal and whitespace cleanup on text. */
  static String cleanText(String text) {
    if (text == null) {
      throw new NullPointerException("The input String is null.");
    }

    StringBuilder stringBuilder = new StringBuilder("");
    for (int index = 0; index < text.length(); index++) {
      char ch = text.charAt(index);

      // Skip the characters that cannot be used.
      if (CharChecker.isInvalid(ch) || CharChecker.isControl(ch)) {
        continue;
      }
      if (CharChecker.isWhitespace(ch)) {
        stringBuilder.append(" ");
      } else {
        stringBuilder.append(ch);
      }
    }
    return stringBuilder.toString();
  }

  /* Runs basic whitespace cleaning and splitting on a piece of text. */
  static List<String> whitespaceTokenize(String text) {
    if (text == null) {
      throw new NullPointerException("The input String is null.");
    }
    return Arrays.asList(text.split(" "));
  }

  /* Splits punctuation on a piece of text. */
  static List<String> runSplitOnPunc(String text) {
    if (text == null) {
      throw new NullPointerException("The input String is null.");
    }

    List<String> tokens = new ArrayList<>();
    boolean startNewWord = true;
    for (int i = 0; i < text.length(); i++) {
      char ch = text.charAt(i);
      if (CharChecker.isPunctuation(ch)) {
        tokens.add(String.valueOf(ch));
        startNewWord = true;
      } else {
        if (startNewWord) {
          tokens.add("");
          startNewWord = false;
        }
        tokens.set(tokens.size() - 1, Iterables.getLast(tokens) + ch);
      }
    }

    return tokens;
  }
}
