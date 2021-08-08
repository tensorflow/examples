package org.tensorflow.lite.examples.poseestimation

import android.content.Context
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import junit.framework.TestCase.assertEquals
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.tensorflow.lite.examples.poseestimation.data.Device
import org.tensorflow.lite.examples.poseestimation.ml.ModelType
import org.tensorflow.lite.examples.poseestimation.ml.MoveNet
import org.tensorflow.lite.examples.poseestimation.ml.PoseClassifier
import org.tensorflow.lite.examples.poseestimation.ml.PoseDetector


@RunWith(AndroidJUnit4::class)
class PoseClassifierTest {

    companion object {
        private const val TEST_INPUT_IMAGE = "image3.jpeg"
    }

    private lateinit var appContext: Context
    private lateinit var poseDetector: PoseDetector
    private lateinit var poseClassifier: PoseClassifier

    @Before
    fun setup() {
        appContext = InstrumentationRegistry.getInstrumentation().targetContext
        poseDetector = MoveNet.create(appContext, Device.CPU, ModelType.Lightning)
        poseClassifier = PoseClassifier.create(appContext)
    }

    @Test
    fun testPoseClassifier() {
        val input = EvaluationUtils.loadBitmapAssetByName(TEST_INPUT_IMAGE)
        // As Movenet use previous frame to optimize detection result, we run it multiple times
        // using the same image to improve result.
        poseDetector.estimateSinglePose(input)
        poseDetector.estimateSinglePose(input)
        poseDetector.estimateSinglePose(input)
        val person = poseDetector.estimateSinglePose(input)
        val classificationResult = poseClassifier.classify(person)
        val predictedPose = classificationResult.maxByOrNull { it.second }?.first ?: "n/a"
        assertEquals(
            "Predicted pose is different from ground truth.",
            "tree",
            predictedPose
        )
    }
}
