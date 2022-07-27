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

package org.tensorflow.lite.examples.imagesegmentation

import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.google.common.truth.Truth.assertThat
import org.junit.Test
import org.junit.runner.RunWith
import org.tensorflow.lite.task.vision.segmenter.Segmentation
import java.io.InputStream

/**
 * Instrumented test, which will execute on an Android device.
 *
 * See [testing documentation](http://d.android.com/tools/testing).
 */
@RunWith(AndroidJUnit4::class)
class ImageSegmentationHelperTest {
    companion object {
        private const val EXPECTED_MASK_TOLERANCE = 1e-2
        private const val INPUT_IMAGE = "input_image.jpg"
        private const val EXPECTED_IMAGE = "expect_image.png"
        private val expectedLabelArray = arrayOf("background", "person", "horse")
    }

    @Test
    fun executeResultShouldNotChange() {
        val imageSegmentationHelper = ImageSegmentationHelper(
            context = InstrumentationRegistry.getInstrumentation().context,
            imageSegmentationListener = object : ImageSegmentationHelper.SegmentationListener {
                override fun onError(error: String) {
                    // no op
                }

                override fun onResults(
                    results: List<Segmentation>?,
                    inferenceTime: Long,
                    imageHeight: Int,
                    imageWidth: Int
                ) {
                    assert(results != null && results.isNotEmpty())

                    val colorLabels = results!![0].coloredLabels.mapIndexed { index, coloredLabel ->
                        OverlayView.ColorLabel(
                            index,
                            coloredLabel.getlabel(),
                            coloredLabel.argb
                        )
                    }

                    // Create the mask bitmap with colors and the set of detected labels.
                    val maskTensor = results[0].masks[0]
                    val maskArray = maskTensor.buffer.array()
                    val pixels = IntArray(maskArray.size)

                    for (i in maskArray.indices) {
                        // Set isExist flag to true if any pixel contains this color.
                        val colorLabel = colorLabels[maskArray[i].toInt()].apply {
                            isExist = true
                        }
                        val color = colorLabel.getColor()
                        pixels[i] = color
                    }

                    val maskImage = Bitmap.createBitmap(
                        pixels,
                        maskTensor.width,
                        maskTensor.height,
                        Bitmap.Config.ARGB_8888
                    )

                    // Verify output mask bitmap.
                    val expectedPixels = getPixels(loadImage(EXPECTED_IMAGE)!!)
                    val resultPixels = getPixels(maskImage)

                    assertThat(resultPixels.size).isEqualTo(expectedPixels.size)

                    var inconsistentPixels = 0
                    for (i in resultPixels.indices) {
                        if (resultPixels[i] != expectedPixels[i]) {
                            inconsistentPixels++
                        }
                    }

                    assertThat(inconsistentPixels.toDouble() / resultPixels.size)
                        .isLessThan(EXPECTED_MASK_TOLERANCE)

                    // Verify labels.
                    val resultLabels = HashSet<String>()
                    colorLabels.filter { it.isExist }.forEach {
                        resultLabels.add(it.label)
                    }

                    val expectedLabels = HashSet<String>()
                    expectedLabelArray.forEach {
                        expectedLabels.add(it)
                    }

                    assertThat(resultLabels).isEqualTo(expectedLabels)
                }
            })
        loadImage(INPUT_IMAGE)?.let {
            imageSegmentationHelper.segment(it, 0)
        }
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
