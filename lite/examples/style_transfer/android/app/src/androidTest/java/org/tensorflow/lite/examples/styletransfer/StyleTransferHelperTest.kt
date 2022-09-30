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

package org.tensorflow.lite.examples.styletransfer

import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.google.common.truth.Truth.assertThat

import org.junit.Test
import org.junit.runner.RunWith

import java.io.InputStream

/**
 * Instrumented test, which will execute on an Android device.
 *
 * See [testing documentation](http://d.android.com/tools/testing).
 */
@RunWith(AndroidJUnit4::class)
class StyleTransferHelperTest {
    companion object {
        private const val INPUT_IMAGE = "input_image.jpg"
        private const val INPUT_STYLE = "input_style.jpg"
        private const val EXPECTED_IMAGE = "expected_image.png"
        private const val EXPECTED_OUTPUT_TOLERANCE = 1e-2
    }

    @Test
    fun executeResultShouldNotChange() {
        val styleTransferHelper =
            StyleTransferHelper(context = InstrumentationRegistry
                .getInstrumentation().context,
                styleTransferListener = object
                    : StyleTransferHelper.StyleTransferListener {
                    override fun onError(error: String) {
                        // no-op
                    }

                    override fun onResult(bitmap: Bitmap, inferenceTime: Long) {

                        // Verify output bitmap.
                        val expectedPixels =
                            getPixels(loadImage(EXPECTED_IMAGE)!!)
                        val resultPixels = getPixels(bitmap)
                        assertThat(resultPixels.size).isEqualTo(expectedPixels.size)
                        var inconsistentPixels = 0
                        for (i in resultPixels.indices) {
                            if (resultPixels[i] != expectedPixels[i]) {
                                inconsistentPixels++
                            }
                        }

                        assertThat(inconsistentPixels.toDouble() / resultPixels.size)
                            .isLessThan(EXPECTED_OUTPUT_TOLERANCE)
                    }
                })
        styleTransferHelper.setStyleImage(loadImage(INPUT_STYLE)!!)
        styleTransferHelper.transfer(loadImage(INPUT_IMAGE)!!)
    }

    @Throws(Exception::class)
    private fun loadImage(fileName: String): Bitmap? {
        val assetManager: AssetManager =
            InstrumentationRegistry.getInstrumentation().context.assets
        val inputStream: InputStream = assetManager.open(fileName)
        return BitmapFactory.decodeStream(inputStream)
    }

    private fun getPixels(bitmap: Bitmap): IntArray {
        val width = bitmap.width
        val height = bitmap.height
        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)
        return pixels
    }
}
