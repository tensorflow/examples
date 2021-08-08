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
import org.tensorflow.lite.examples.poseestimation.ml.ModelType
import org.tensorflow.lite.examples.poseestimation.ml.MoveNet
import org.tensorflow.lite.examples.poseestimation.ml.PoseDetector


@RunWith(AndroidJUnit4::class)
class MovenetLightningTest {

    companion object {

        private const val TEST_INPUT_IMAGE1 = "image1"
        private val EXPECTED_DETECTION_RESULT1 = mapOf(
            BodyPart.NOSE to PointF(193.0462f, 87.497574f),
            BodyPart.LEFT_EYE to PointF(209.29642f, 75.67456f),
            BodyPart.RIGHT_EYE to PointF(182.6607f, 78.23213f),
            BodyPart.LEFT_EAR to PointF(239.74228f, 88.43133f),
            BodyPart.RIGHT_EAR to PointF(176.84341f, 89.485374f),
            BodyPart.LEFT_SHOULDER to PointF(253.89224f, 162.15315f),
            BodyPart.RIGHT_SHOULDER to PointF(152.12976f, 155.90091f),
            BodyPart.LEFT_ELBOW to PointF(270.097f, 260.88635f),
            BodyPart.RIGHT_ELBOW to PointF(148.23059f, 239.923f),
            BodyPart.LEFT_WRIST to PointF(275.47607f, 335.0756f),
            BodyPart.RIGHT_WRIST to PointF(142.26117f, 311.81918f),
            BodyPart.LEFT_HIP to PointF(238.68332f, 329.58127f),
            BodyPart.RIGHT_HIP to PointF(178.08572f, 331.83063f),
            BodyPart.LEFT_KNEE to PointF(260.20868f, 468.5389f),
            BodyPart.RIGHT_KNEE to PointF(141.22626f, 467.30423f),
            BodyPart.LEFT_ANKLE to PointF(273.98502f, 588.24274f),
            BodyPart.RIGHT_ANKLE to PointF(95.03668f, 597.6913f),
        )

        private const val TEST_INPUT_IMAGE2 = "image2"
        private val EXPECTED_DETECTION_RESULT2 = mapOf(
            BodyPart.NOSE to PointF(185.01096f, 86.7739f),
            BodyPart.LEFT_EYE to PointF(193.2121f, 75.5961f),
            BodyPart.RIGHT_EYE to PointF(172.3854f, 76.547386f),
            BodyPart.LEFT_EAR to PointF(204.05804f, 77.61157f),
            BodyPart.RIGHT_EAR to PointF(156.31363f, 78.961266f),
            BodyPart.LEFT_SHOULDER to PointF(219.9895f, 125.02336f),
            BodyPart.RIGHT_SHOULDER to PointF(144.1854f, 131.37856f),
            BodyPart.LEFT_ELBOW to PointF(259.59085f, 197.88562f),
            BodyPart.RIGHT_ELBOW to PointF(180.91986f, 214.5548f),
            BodyPart.LEFT_WRIST to PointF(247.00491f, 214.88852f),
            BodyPart.RIGHT_WRIST to PointF(233.76907f, 212.72563f),
            BodyPart.LEFT_HIP to PointF(219.44794f, 289.7696f),
            BodyPart.RIGHT_HIP to PointF(176.40805f, 293.85168f),
            BodyPart.LEFT_KNEE to PointF(206.05576f, 421.18146f),
            BodyPart.RIGHT_KNEE to PointF(173.7746f, 426.6271f),
            BodyPart.LEFT_ANKLE to PointF(188.79883f, 534.07745f),
            BodyPart.RIGHT_ANKLE to PointF(157.41333f, 566.5951f),
        )
    }

    private lateinit var poseDetector: PoseDetector
    private lateinit var appContext: Context

    @Before
    fun setup() {
        appContext = InstrumentationRegistry.getInstrumentation().targetContext
        poseDetector = MoveNet.create(appContext, Device.CPU, ModelType.Lightning)
    }

    @Test
    fun testPoseEstimationResultWithImage1() {
        val input = EvaluationUtils.loadBitmapResourceByName(TEST_INPUT_IMAGE1)

        // As Movenet use previous frame to optimize detection result, we run it multiple times
        // using the same image to improve result.
        poseDetector.estimateSinglePose(input)
        poseDetector.estimateSinglePose(input)
        poseDetector.estimateSinglePose(input)
        val person = poseDetector.estimateSinglePose(input)
        EvaluationUtils.assertPoseDetectionResult(person, EXPECTED_DETECTION_RESULT1)
    }

    @Test
    fun testPoseEstimationResultWithImage2() {
        val input = EvaluationUtils.loadBitmapResourceByName(TEST_INPUT_IMAGE2)

        // As Movenet use previous frame to optimize detection result, we run it multiple times
        // using the same image to improve result.
        poseDetector.estimateSinglePose(input)
        poseDetector.estimateSinglePose(input)
        poseDetector.estimateSinglePose(input)
        val person = poseDetector.estimateSinglePose(input)
        EvaluationUtils.assertPoseDetectionResult(person, EXPECTED_DETECTION_RESULT2)
    }
}