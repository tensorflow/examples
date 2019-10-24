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

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/** Word piece tokenization to split a piece of text into its word pieces. */
public final class WordpieceTokenizer {
  private final Map<String, Integer> dic;

  private static final String UNKNOWN_TOKEN = "[UNK]"; // For unknown words.
  private static final int MAX_INPUTCHARS_PER_WORD = 200;

  public WordpieceTokenizer(Map<String, Integer> vocab) {
    dic = vocab;
  }

  /**
   * Tokenizes a piece of text into its word pieces. This uses a greedy longest-match-first
   * algorithm to perform tokenization using the given vocabulary. For example: input = "unaffable",
   * output = ["un", "##aff", "##able"].
   *
   * @param text: A single token or whitespace separated tokens. This should have already been
   *     passed through `BasicTokenizer.
   * @return A list of wordpiece tokens.
   */
  public List<String> tokenize(String text) {
    if (text == null) {
      throw new NullPointerException("The input String is null.");
    }

    List<String> outputTokens = new ArrayList<>();
    for (String token : BasicTokenizer.whitespaceTokenize(text)) {

      if (token.length() > MAX_INPUTCHARS_PER_WORD) {
        outputTokens.add(UNKNOWN_TOKEN);
        continue;
      }

      boolean isBad = false; // Mark if a word cannot be tokenized into known subwords.
      int start = 0;
      List<String> subTokens = new ArrayList<>();

      while (start < token.length()) {
        String curSubStr = "";

        int end = token.length(); // Longer substring matches first.
        while (start < end) {
          String subStr =
              (start == 0) ? token.substring(start, end) : "##" + token.substring(start, end);
          if (dic.containsKey(subStr)) {
            curSubStr = subStr;
            break;
          }
          end--;
        }

        // The word doesn't contain any known subwords.
        if ("".equals(curSubStr)) {
          isBad = true;
          break;
        }

        // curSubStr is the longeset subword that can be found.
        subTokens.add(curSubStr);

        // Proceed to tokenize the resident string.
        start = end;
      }

      if (isBad) {
        outputTokens.add(UNKNOWN_TOKEN);
      } else {
        outputTokens.addAll(subTokens);
      }
    }

    return outputTokens;
  }
}
