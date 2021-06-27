package org.tensorflow.lite.examples.poseestimation

import android.content.Context
import android.graphics.PointF
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.tensorflow.lite.examples.poseestimation.data.BodyPart
import org.tensorflow.lite.examples.poseestimation.data.Device
import org.tensorflow.lite.examples.poseestimation.ml.PoseDetector
import org.tensorflow.lite.examples.poseestimation.ml.PoseNet


@RunWith(AndroidJUnit4::class)
class PosenetTest {

    companion object {
        private const val TEST_INPUT_IMAGE1 = "image1"
        private val EXPECTED_DETECTION_RESULT1 = mapOf(
            BodyPart.NOSE to PointF(192f, 121f),
            BodyPart.LEFT_EYE to PointF(200f, 112f),
            BodyPart.RIGHT_EYE to PointF(189f, 112f),
            BodyPart.LEFT_EAR to PointF(221f, 113f),
            BodyPart.RIGHT_EAR to PointF(186f, 114f),
            BodyPart.LEFT_SHOULDER to PointF(244f, 159f),
            BodyPart.RIGHT_SHOULDER to PointF(166f, 158f),
            BodyPart.LEFT_ELBOW to PointF(267f, 252f),
            BodyPart.RIGHT_ELBOW to PointF(157f, 248f),
            BodyPart.LEFT_WRIST to PointF(276f, 336f),
            BodyPart.RIGHT_WRIST to PointF(147f, 326f),
            BodyPart.LEFT_HIP to PointF(246f, 322f),
            BodyPart.RIGHT_HIP to PointF(174f, 325f),
            BodyPart.LEFT_KNEE to PointF(261f, 466f),
            BodyPart.RIGHT_KNEE to PointF(143f, 465f),
            BodyPart.LEFT_ANKLE to PointF(273f, 548f),
            BodyPart.RIGHT_ANKLE to PointF(130f, 550f),
        )

        private const val TEST_INPUT_IMAGE2 = "image2"
        private val EXPECTED_DETECTION_RESULT2 = mapOf(
            BodyPart.NOSE to PointF(192f, 102f),
            BodyPart.LEFT_EYE to PointF(197f, 97f),
            BodyPart.RIGHT_EYE to PointF(190f, 98f),
            BodyPart.LEFT_EAR to PointF(200f, 90f),
            BodyPart.RIGHT_EAR to PointF(178f, 94f),
            BodyPart.LEFT_SHOULDER to PointF(218f, 124f),
            BodyPart.RIGHT_SHOULDER to PointF(147f, 134f),
            BodyPart.LEFT_ELBOW to PointF(265f, 197f),
            BodyPart.RIGHT_ELBOW to PointF(180f, 220f),
            BodyPart.LEFT_WRIST to PointF(239f, 217f),
            BodyPart.RIGHT_WRIST to PointF(232f, 217f),
            BodyPart.LEFT_HIP to PointF(220f, 310f),
            BodyPart.RIGHT_HIP to PointF(173f, 287f),
            BodyPart.LEFT_KNEE to PointF(213f, 425f),
            BodyPart.RIGHT_KNEE to PointF(174f, 425f),
            BodyPart.LEFT_ANKLE to PointF(184f, 531f),
            BodyPart.RIGHT_ANKLE to PointF(161f, 540f),
        )
    }

    private lateinit var poseDetector: PoseDetector
    private lateinit var appContext: Context

    @Before
    fun setup() {
        appContext = InstrumentationRegistry.getInstrumentation().targetContext
        poseDetector = PoseNet.create(appContext, Device.CPU)
    }

    @Test
    fun testPoseEstimationResultWithImage1() {
        val input = EvaluationUtils.loadBitmapResourceByName(TEST_INPUT_IMAGE1)
        val person = poseDetector.estimateSinglePose(input)
        EvaluationUtils.assertPoseDetectionResult(person, EXPECTED_DETECTION_RESULT1)
    }

    @Test
    fun testPoseEstimationResultWithImage2() {
        val input = EvaluationUtils.loadBitmapResourceByName(TEST_INPUT_IMAGE2)
        val person = poseDetector.estimateSinglePose(input)
        EvaluationUtils.assertPoseDetectionResult(person, EXPECTED_DETECTION_RESULT2)
    }
}