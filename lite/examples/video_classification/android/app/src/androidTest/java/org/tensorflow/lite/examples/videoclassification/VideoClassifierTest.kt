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

package org.tensorflow.lite.examples.videoclassification

import android.content.Context
import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.hamcrest.MatcherAssert.assertThat
import org.hamcrest.Matchers.greaterThan
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.tensorflow.lite.examples.videoclassification.ml.VideoClassifier
import org.tensorflow.lite.support.label.Category
import java.io.*

@RunWith(AndroidJUnit4::class)
class VideoClassificationTest {

    companion object {
        private const val INPUT_FILE_NAME = "carving_ice.mp4"
    }

    private lateinit var videoClassifier: VideoClassifier
    private lateinit var appContext: Context
    private lateinit var testContext: Context

    @Before
    fun setup() {
        appContext = InstrumentationRegistry.getInstrumentation().targetContext
        testContext = InstrumentationRegistry.getInstrumentation().context
        val options =
            VideoClassifier.VideoClassifierOptions.builder().setMaxResult(5)
                .setNumThreads(1)
                .build()
        videoClassifier = VideoClassifier.createFromFileAndLabelsAndOptions(
            appContext,
            "movinet_a0_stream_int8.tflite",
            "kinetics600_label_map.txt",
            options
        )
    }

    @Test
    fun testVideoClassifierWithExampleVideo() {
        // Load the test video.
        val inputStream = testContext.assets.open(INPUT_FILE_NAME)
        val file = createFileFromInputStream(inputStream)
        val media = MediaMetadataRetriever()
        media.setDataSource(file.path)

        // Run classification on all frames in the video
        var categories: List<Category> = listOf()
        val frameCount = media.extractMetadata(
            MediaMetadataRetriever.METADATA_KEY_VIDEO_FRAME_COUNT)!!.toInt()
        for (i in 0 until frameCount) {
            val inputFrame = media
                .getFrameAtIndex(i)
                ?.copy(Bitmap.Config.ARGB_8888, true)
            inputFrame?.let {
                categories = videoClassifier.classify(it)
            }
        }
        media.release()

        // Assert if the top-1 classification result matches with ground truth.
        assertTrue("Classification result isn't empty.", categories.isNotEmpty())
        assertEquals("Top1 category matches.", "carving ice", categories[0].label)
        assertThat("Score is larger than threshold.", categories[0].score, greaterThan(0.6f))
    }

    private fun createFileFromInputStream(inputStream: InputStream): File {
        val f = File("${appContext.cacheDir}/$INPUT_FILE_NAME")
        val outputStream: OutputStream = FileOutputStream(f)
        val buffer = ByteArray(1024)
        var length: Int
        while (inputStream.read(buffer).also { length = it } > 0) {
            outputStream.write(buffer, 0, length)
        }
        outputStream.close()
        inputStream.close()
        return f
    }
}
