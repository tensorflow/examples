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
class MovenetThunderTest {

    companion object {

        private const val TEST_INPUT_IMAGE1 = "image1"
        private val EXPECTED_DETECTION_RESULT1 = mapOf(
            BodyPart.NOSE to PointF(191.30023f, 91.054016f),
            BodyPart.LEFT_EYE to PointF(207.9235f, 78.95678f),
            BodyPart.RIGHT_EYE to PointF(184.22388f, 79.73614f),
            BodyPart.LEFT_EAR to PointF(234.63568f, 91.59316f),
            BodyPart.RIGHT_EAR to PointF(182.32257f, 86.59598f),
            BodyPart.LEFT_SHOULDER to PointF(253.46219f, 155.86337f),
            BodyPart.RIGHT_SHOULDER to PointF(152.81116f, 148.50366f),
            BodyPart.LEFT_ELBOW to PointF(269.09348f, 257.75925f),
            BodyPart.RIGHT_ELBOW to PointF(154.00763f, 245.15147f),
            BodyPart.LEFT_WRIST to PointF(275.6372f, 341.65726f),
            BodyPart.RIGHT_WRIST to PointF(145.23264f, 322.2945f),
            BodyPart.LEFT_HIP to PointF(242.05112f, 327.1707f),
            BodyPart.RIGHT_HIP to PointF(180.30011f, 326.49484f),
            BodyPart.LEFT_KNEE to PointF(259.78342f, 463.29733f),
            BodyPart.RIGHT_KNEE to PointF(142.21649f, 466.35083f),
            BodyPart.LEFT_ANKLE to PointF(272.93997f, 585.4292f),
            BodyPart.RIGHT_ANKLE to PointF(95.16376f, 591.59485f),
        )

        private const val TEST_INPUT_IMAGE2 = "image2"
        private val EXPECTED_DETECTION_RESULT2 = mapOf(
            BodyPart.NOSE to PointF(181.48196f, 85.54128f),
            BodyPart.LEFT_EYE to PointF(191.36609f, 73.53658f),
            BodyPart.RIGHT_EYE to PointF(170.82925f, 74.48267f),
            BodyPart.LEFT_EAR to PointF(203.28955f, 74.87907f),
            BodyPart.RIGHT_EAR to PointF(158.07617f, 78.84626f),
            BodyPart.LEFT_SHOULDER to PointF(223.98969f, 119.73586f),
            BodyPart.RIGHT_SHOULDER to PointF(139.46057f, 137.01653f),
            BodyPart.LEFT_ELBOW to PointF(264.64807f, 194.49603f),
            BodyPart.RIGHT_ELBOW to PointF(174.77829f, 221.04686f),
            BodyPart.LEFT_WRIST to PointF(258.24692f, 215.45662f),
            BodyPart.RIGHT_WRIST to PointF(234.78876f, 223.67337f),
            BodyPart.LEFT_HIP to PointF(227.87497f, 288.54553f),
            BodyPart.RIGHT_HIP to PointF(176.99933f, 291.8097f),
            BodyPart.LEFT_KNEE to PointF(206.76163f, 422.22635f),
            BodyPart.RIGHT_KNEE to PointF(172.21161f, 423.68784f),
            BodyPart.LEFT_ANKLE to PointF(182.44028f, 546.49677f),
            BodyPart.RIGHT_ANKLE to PointF(149.3563f, 565.5011f),
        )
    }

    private lateinit var poseDetector: PoseDetector
    private lateinit var appContext: Context

    @Before
    fun setup() {
        appContext = InstrumentationRegistry.getInstrumentation().targetContext
        poseDetector = MoveNet.create(appContext, Device.CPU, ModelType.Thunder)
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