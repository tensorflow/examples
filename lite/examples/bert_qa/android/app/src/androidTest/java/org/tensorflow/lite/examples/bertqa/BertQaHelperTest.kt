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
package org.tensorflow.lite.examples.bertqa

import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.ext.junit.runners.AndroidJUnit4

import org.junit.Test
import org.junit.runner.RunWith

import org.junit.Assert.*
import org.junit.Before
import org.tensorflow.lite.task.text.qa.QaAnswer

/**
 * Instrumented test, which will execute on an Android device.
 *
 * See [testing documentation](http://d.android.com/tools/testing).
 */
@RunWith(AndroidJUnit4::class)
class BertQaHelperTest {
    private val content =
        "Doraemon is a blue cat automaton corresponding (tints of pink-orange in earlier comic chapters and media) from the 22nd century, who weighs 129.3kg (285.05lbs) and measures at 129.3cm (4'3\") tall."
    private val question = "What color is Doraemon?"
    private val answer = "blue"

    private lateinit var helper: BertQaHelper

    @Test
    fun bertAnswersShouldNotChange() {
        helper = BertQaHelper(context = InstrumentationRegistry.getInstrumentation().context,
            answererListener = object : BertQaHelper.AnswererListener {
                override fun onError(error: String) {
                    // no op
                }

                override fun onResults(results: List<QaAnswer>?, inferenceTime: Long) {
                    assert(!results.isNullOrEmpty())

                    // verify bert qa answer and expected answer.
                    assertEquals(answer, results!![0].text)
                }
            })
        helper.answer(content, question)
    }
}
