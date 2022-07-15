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
package org.tensorflow.lite.examples.textclassification

import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.ext.junit.runners.AndroidJUnit4

import org.junit.Test
import org.junit.runner.RunWith

import org.junit.Assert.*
import org.tensorflow.lite.support.label.Category

/**
 * Instrumented test, which will execute on an Android device.
 *
 * See [testing documentation](http://d.android.com/tools/testing).
 */
@RunWith(AndroidJUnit4::class)
class TextClassifierInstrumentationTest {

    val testText = "This was a triumph. I\\'m making a note here, HUGE SUCCESS. It\\'s hard to " +
            "overstate my satisfaction."

    val nlClassifierPositiveConfidence = 0.5736692f
    val nlClassifierNegativeConfidence = 0.4263308f

    @Test
    fun TextClassificationHelperReturnsConsistentConfidenceResults() {
        val textClassificationHelper =
            TextClassificationHelper(
                context = InstrumentationRegistry.getInstrumentation().context,
                listener = object : TextClassificationHelper.TextResultsListener {
                    override fun onError(error: String) {
                        // no op
                    }

                    override fun onResult(results: List<Category>, inferenceTime: Long) {
                        assertEquals(results[0].score, nlClassifierNegativeConfidence)
                        assertEquals(results[1].score, nlClassifierPositiveConfidence)
                    }
                }
            )

        textClassificationHelper.classify(testText)
    }
}