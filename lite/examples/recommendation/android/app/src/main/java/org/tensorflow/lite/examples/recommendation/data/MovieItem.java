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
package org.tensorflow.lite.examples.recommendation.data;

import android.text.TextUtils;
import java.util.ArrayList;
import java.util.List;

/** A movie item representing recommended content. */
public class MovieItem {

  public static final String JOINER = " | ";
  public static final String DELIMITER_REGEX = "[|]";

  public final int id;
  public final String title;
  public final List<String> genres;
  public final int count;

  public boolean selected = false; // For UI selection. Default item is not selected.

  private MovieItem() {
    this(0, "", new ArrayList<>(), 0);
  }

  public MovieItem(int id, String title, List<String> genres, int count) {
    this.id = id;
    this.title = title;
    this.genres = genres;
    this.count = count;
  }

  @Override
  public String toString() {
    return String.format(
        "Id: %d, title: %s, genres: %s, count: %d, selected: %s",
        id, title, TextUtils.join(JOINER, genres), count, selected);
  }

}
