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
package org.tensorflow.lite.examples.imageclassification

import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.view.Surface
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.ext.junit.runners.AndroidJUnit4

import org.junit.Test
import org.junit.runner.RunWith

import org.junit.Assert.*
import org.tensorflow.lite.support.label.Category
import org.tensorflow.lite.task.vision.classifier.Classifications
import java.io.InputStream
import java.lang.Exception

/**
 * Instrumented test, which will execute on an Android device.
 *
 * See [testing documentation](http://d.android.com/tools/testing).
 */
@RunWith(AndroidJUnit4::class)
class ImageClassificationTest {

    val controlCategories = listOf<Category>(
        Category.create("cup", "cup", 0.76953125f),
        Category.create("coffee mug", "coffee mug", 0.125f),
        Category.create("espresso", "espresso", 0.0546875f)
    )

    @Test
    @Throws(Exception::class)
    fun classificationResultsShouldNotChange() {
        val imageClassifierHelper = ImageClassifierHelper(
            context = InstrumentationRegistry.getInstrumentation().context,
            imageClassifierListener = object : ImageClassifierHelper.ClassifierListener {
                override fun onError(error: String) {
                    // no op
                }

                override fun onResults(
                    results: List<Classifications>?,
                    inferenceTime: Long
                ) {
                    assertNotNull(results)

                    // Verify that the classified data and control
                    // data have the same number of categories
                    assertEquals(controlCategories.size, results!![0].categories.size)

                    // Loop through the categories
                    for (i in controlCategories.indices) {
                        // Verify that the labels are consistent
                        assertEquals(controlCategories[i].label, results[0].categories[i].label)
                    }
                }
            }, threshold = 0.0f
        )

        // Create Bitmap and convert to TensorImage
        val bitmap = loadImage("coffee.jpg")
        // Run the image classifier on the sample image. Assume device portrait orientation.
        imageClassifierHelper.classify(bitmap!!, Surface.ROTATION_90)
    }

    @Throws(Exception::class)
    private fun loadImage(fileName: String): Bitmap? {
        val assetManager: AssetManager =
            InstrumentationRegistry.getInstrumentation().context.assets
        val inputStream: InputStream = assetManager.open(fileName)
        return BitmapFactory.decodeStream(inputStream)
    }
}
