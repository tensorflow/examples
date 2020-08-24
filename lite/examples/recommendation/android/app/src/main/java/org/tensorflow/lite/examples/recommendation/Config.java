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

package org.tensorflow.lite.examples.recommendation;

/** Config for recommendation app. */
public final class Config {
  private static final String DEFAULT_MODEL_PATH = "recommendation_rnn_i10o100.tflite";
  private static final String DEFAULT_MOVIE_LIST_PATH = "sorted_movie_vocab.json";
  private static final int DEFAULT_INPUT_LENGTH = 10;
  private static final int DEFAULT_OUTPUT_LENGTH = 100;
  private static final int DEFAULT_TOP_K = 10;
  private static final int PAD_ID = 0;
  private static final int DEFAULT_OUTPUT_IDS_INDEX = 0;
  private static final int DEFAULT_OUTPUT_SCORES_INDEX = 1;
  private static final int DEFAULT_FAVORITE_LIST_SIZE = 100;

  /** TF Lite model path. */
  public String modelPath = DEFAULT_MODEL_PATH;
  /** Number of input ids from the model. */
  public int inputLength = DEFAULT_INPUT_LENGTH;
  /** Number of output length from the model. */
  public int outputLength = DEFAULT_OUTPUT_LENGTH;
  /** Number of max results to show in the UI. */
  public int topK = DEFAULT_TOP_K;
  /** Path to the movie list. */
  public String movieListPath = DEFAULT_MOVIE_LIST_PATH;

  /** Id for padding. */
  public int pad = PAD_ID;
  /** Output index for ID. */
  public int outputIdsIndex = DEFAULT_OUTPUT_IDS_INDEX;
  /** Output index for score. */
  public int outputScoresIndex = DEFAULT_OUTPUT_SCORES_INDEX;

  /** The number of favorite movies for users to choose from. */
  public int favoriteListSize = DEFAULT_FAVORITE_LIST_SIZE;

  public Config() {}
}
