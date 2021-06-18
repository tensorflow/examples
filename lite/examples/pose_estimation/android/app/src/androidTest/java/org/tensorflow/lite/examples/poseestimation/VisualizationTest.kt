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
import org.tensorflow.lite.examples.poseestimation.ml.PoseDetector
import org.tensorflow.lite.examples.poseestimation.ml.PoseNet

@RunWith(AndroidJUnit4::class)
class VisualizationTest {

    companion object {
        private const val TEST_INPUT_IMAGE = "image1"
    }

    private lateinit var appContext: Context
    private lateinit var inputBitmap: Bitmap

    @Before
    fun setup() {
        appContext = InstrumentationRegistry.getInstrumentation().targetContext
        inputBitmap = EvaluationUtils.loadBitmapResourceByName(TEST_INPUT_IMAGE)
    }

    /**
     * This test is used to visually verify detection results by the models.
     * You can put a breakpoint at the end of the method, debug this method, than use the
     * "View Bitmap" feature of the debugger to check the visualized detection result.
     * This test can also be used to generate expected result for the detection tests.
     * See goto.google.com/tflite-pose-estimation-testgen for details.
     */
    @Test
    fun testVisualization() {
        var poseDetector: PoseDetector

        // Visualization for Posenet
        poseDetector = PoseNet.create(appContext, Device.CPU)
        val person1 = poseDetector.estimateSinglePose(inputBitmap)
        val outputBitmap1 = VisualizationUtils.drawBodyKeypoints(inputBitmap, person1)
        assertThat(outputBitmap1).isNotNull()

        // Visualization for Movenet Lightning
        poseDetector = MoveNet.create(appContext, Device.CPU, ModelType.Lightning)
        poseDetector.estimateSinglePose(inputBitmap)
        poseDetector.estimateSinglePose(inputBitmap)
        val person2 = poseDetector.estimateSinglePose(inputBitmap)
        val outputBitmap2 = VisualizationUtils.drawBodyKeypoints(inputBitmap, person2)
        assertThat(outputBitmap2).isNotNull()

        // Visualization for Movenet Thunder
        poseDetector = MoveNet.create(appContext, Device.CPU, ModelType.Thunder)
        poseDetector.estimateSinglePose(inputBitmap)
        poseDetector.estimateSinglePose(inputBitmap)
        val person3 = poseDetector.estimateSinglePose(inputBitmap)
        val outputBitmap3 = VisualizationUtils.drawBodyKeypoints(inputBitmap, person3)
        assertThat(outputBitmap3).isNotNull()
    }
}