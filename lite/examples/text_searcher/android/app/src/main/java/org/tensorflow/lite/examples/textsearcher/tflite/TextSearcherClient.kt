/*
 * Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

package org.tensorflow.lite.examples.textsearcher.tflite

import android.content.Context
import java.io.Serializable
import org.tensorflow.lite.task.core.BaseOptions
import org.tensorflow.lite.task.processor.SearcherOptions
import org.tensorflow.lite.task.text.searcher.TextSearcher

class TextSearcherClient(private var textSearcher: TextSearcher) {

  companion object {
    private const val MODEL_PATH = "text_searcher.tflite"
    private const val NUM_THREADS = 4
    private const val MAX_RESULTS = 10
    private const val IS_NORMALIZE = true

    fun create(context: Context): TextSearcherClient {
      val baseOptions = BaseOptions.builder().setNumThreads(NUM_THREADS).build()
      val searchOptions =
        SearcherOptions.builder().setMaxResults(MAX_RESULTS).setL2Normalize(IS_NORMALIZE).build()
      val options =
        TextSearcher.TextSearcherOptions.builder()
          .setBaseOptions(baseOptions)
          .setSearcherOptions(searchOptions)
          .build()
      val textSearcher = TextSearcher.createFromFileAndOptions(context, MODEL_PATH, options)
      return TextSearcherClient(textSearcher)
    }
  }

  fun search(query: String): List<Result> {
    val results = mutableListOf<Result>()
    val modelResults = textSearcher.search(query)

    // Postprocess the model output to human readable class names
    modelResults.forEach { results.add(Result(it.distance, String(it.metadata.array()))) }
    return results
  }

  fun close() {
    textSearcher.close()
  }
}

data class Result(val distance: Float, val url: String) : Serializable
