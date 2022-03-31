/*
 * Copyright 2022 The TensorFlow Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.classification.playservices

import android.content.Context
import android.graphics.BitmapFactory
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.google.android.gms.tasks.Tasks
import com.google.android.gms.tflite.java.TfLite
import com.google.common.truth.Truth.assertThat
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class InstrumentationTest {

  private val maxResult = 3

  @Test
  fun classify_returnKResult() {
    val targetContext = ApplicationProvider.getApplicationContext() as Context
    Tasks.await(TfLite.initialize(targetContext))
    val classifier = createClassifier(targetContext, maxResult)

    val context = InstrumentationRegistry.getInstrumentation().context
    val inputImage = BitmapFactory.decodeStream(context.assets.open("testInput.jpg"))
    val result = classifier.classify(inputImage, 0)
    assertThat(result).hasSize(maxResult)
  }
}
