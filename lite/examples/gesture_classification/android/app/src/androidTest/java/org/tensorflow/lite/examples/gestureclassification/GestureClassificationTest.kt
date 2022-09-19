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
 
package org.tensorflow.lite.examples.gestureclassification

import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert
import org.junit.Test
import org.junit.runner.RunWith
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
class GestureClassificationTest {

    val controlCategories = listOf<Category>(
        Category.create("down", "down", 0.82084465f),
        Category.create("up", "up", 0.08746685f),
        Category.create("scrolldown", "scrolldown", 0.04518125f)
    )

    @Test
    fun classificationResultsShouldNotChange() {
        val gestureClassifierHelper = GestureClassifierHelper(
            context = InstrumentationRegistry.getInstrumentation().context,
            gestureClassifierListener = object :
                GestureClassifierHelper.ClassifierListener {
                override fun onError(error: String) {
                    // no op
                }

                override fun onResults(
                    results: List<Classifications>?,
                    inferenceTime: Long
                ) {
                    Assert.assertNotNull(results)

                    // Verify that the classified data and control
                    // data have the same number of categories
                    Assert.assertEquals(
                        controlCategories.size,
                        results!![0].categories.size
                    )

                    // Loop through the categories
                    for (i in controlCategories.indices) {
                        // Verify that the labels are consistent
                        Assert.assertEquals(
                            controlCategories[i].label,
                            results[0].categories[i].label
                        )
                    }
                }
            }, threshold = 0.0f
        )

        // Create Bitmap and convert to TensorImage
        val bitmap = loadImage("test_image.jpg")
        // Run the gesture classifier on the sample image
        gestureClassifierHelper.classify(bitmap!!, 0)
    }

    @Throws(Exception::class)
    private fun loadImage(fileName: String): Bitmap? {
        val assetManager: AssetManager =
            InstrumentationRegistry.getInstrumentation().context.assets
        val inputStream: InputStream = assetManager.open(fileName)
        return BitmapFactory.decodeStream(inputStream)
    }
}
