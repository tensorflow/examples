/*
 * Copyright 2022 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *             http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.tensorflow.lite.examples.bertqa.dataset

import com.google.gson.annotations.SerializedName

data class DataSet(
    @SerializedName("titles")
    private val titles: List<List<String>>,
    @SerializedName("contents")
    private val contents: List<List<String>>,
    @SerializedName("questions")
    val questions: List<List<String>>
) {
    fun getTitles(): List<String> {
        return titles.map { it[0] }
    }

    fun getContents(): List<String> {
        return contents.map { it[0] }
    }
}
