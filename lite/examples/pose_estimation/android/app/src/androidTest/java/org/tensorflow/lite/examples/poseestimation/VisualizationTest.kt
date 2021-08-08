package org.tensorflow.lite.examples.poseestimation

import android.content.Context
import android.graphics.Bitmap
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.google.common.truth.Truth.assertThat
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.tensorflow.lite.examples.poseestimation.data.Device
import org.tensorflow.lite.examples.poseestimation.ml.ModelType
import org.tensorflow.lite.examples.poseestimation.ml.MoveNet
import org.tensorflow.lite.examples.poseestimation.ml.PoseNet

/**
 * This test is used to visually verify detection results by the models.
 * You can put a breakpoint at the end of the method, debug this method, than use the
 * "View Bitmap" feature of the debugger to check the visualized detection result.
 */
@RunWith(AndroidJUnit4::class)
class VisualizationTest {

    companion object {
        private const val TEST_INPUT_IMAGE = "image2.jpg"
    }

    private lateinit var appContext: Context
    private lateinit var inputBitmap: Bitmap

    @Before
    fun setup() {
        appContext = InstrumentationRegistry.getInstrumentation().targetContext
        inputBitmap = EvaluationUtils.loadBitmapAssetByName(TEST_INPUT_IMAGE)
    }

    @Test
    fun testPosenet() {
        val poseDetector = PoseNet.create(appContext, Device.CPU)
        val person = poseDetector.estimateSinglePose(inputBitmap)
        val outputBitmap = VisualizationUtils.drawBodyKeypoints(inputBitmap, person)
        assertThat(outputBitmap).isNotNull()
    }

    @Test
    fun testMovenetLightning() {
        // Due to Movenet's cropping logic, we run inference several times with the same input
        // image to improve accuracy
        val poseDetector = MoveNet.create(appContext, Device.CPU, ModelType.Lightning)
        poseDetector.estimateSinglePose(inputBitmap)
        poseDetector.estimateSinglePose(inputBitmap)
        val person2 = poseDetector.estimateSinglePose(inputBitmap)
        val outputBitmap2 = VisualizationUtils.drawBodyKeypoints(inputBitmap, person2)
        assertThat(outputBitmap2).isNotNull()
    }

    @Test
    fun testMovenetThunder() {
        // Due to Movenet's cropping logic, we run inference several times with the same input
        // image to improve accuracy
        val poseDetector = MoveNet.create(appContext, Device.CPU, ModelType.Thunder)
        poseDetector.estimateSinglePose(inputBitmap)
        poseDetector.estimateSinglePose(inputBitmap)
        val person = poseDetector.estimateSinglePose(inputBitmap)
        val outputBitmap = VisualizationUtils.drawBodyKeypoints(inputBitmap, person)
        assertThat(outputBitmap).isNotNull()
    }
}