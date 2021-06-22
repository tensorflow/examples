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
            BodyPart.NOSE to PointF(477.55243f,223.51778f),
            BodyPart.LEFT_EYE to PointF(517.18823f,193.21722f),
            BodyPart.RIGHT_EYE to PointF(456.15558f,199.29283f),
            BodyPart.LEFT_EAR to PointF(601.0279f,224.95488f),
            BodyPart.RIGHT_EAR to PointF(456.82843f,222.56891f),
            BodyPart.LEFT_SHOULDER to PointF(641.3848f,411.8017f),
            BodyPart.RIGHT_SHOULDER to PointF(388.57532f,393.41974f),
            BodyPart.LEFT_ELBOW to PointF(676.9551f,664.9355f),
            BodyPart.RIGHT_ELBOW to PointF(376.02216f,615.6095f),
            BodyPart.LEFT_WRIST to PointF(694.2405f,854.1634f),
            BodyPart.RIGHT_WRIST to PointF(357.58026f,800.7173f),
            BodyPart.LEFT_HIP to PointF(600.9557f,833.3673f),
            BodyPart.RIGHT_HIP to PointF(451.46893f,833.7992f),
            BodyPart.LEFT_KNEE to PointF(655.1889f,1180.6663f),
            BodyPart.RIGHT_KNEE to PointF(354.29785f,1181.418f),
            BodyPart.LEFT_ANKLE to PointF(689.87585f,1484.8456f),
        )

        private const val TEST_INPUT_IMAGE2 = "image2"
        private val EXPECTED_DETECTION_RESULT2 = mapOf(
            BodyPart.NOSE to PointF(489.45844f, 227.12387f),
            BodyPart.LEFT_EYE to PointF(509.92798f, 198.13692f),
            BodyPart.RIGHT_EYE to PointF(458.0664f, 199.133f),
            BodyPart.LEFT_EAR to PointF(534.27563f, 201.47113f),
            BodyPart.RIGHT_EAR to PointF(409.5276f, 208.07262f),
            BodyPart.LEFT_SHOULDER to PointF(576.5243f, 327.9697f),
            BodyPart.RIGHT_SHOULDER to PointF(380.95404f, 351.27295f),
            BodyPart.LEFT_ELBOW to PointF(676.0817f, 525.8004f),
            BodyPart.RIGHT_ELBOW to PointF(476.83392f, 563.8921f),
            BodyPart.LEFT_WRIST to PointF(650.15137f, 557.4394f),
            BodyPart.RIGHT_WRIST to PointF(609.00836f, 553.93585f),
            BodyPart.LEFT_HIP to PointF(576.3827f, 763.0239f),
            BodyPart.RIGHT_HIP to PointF(466.8543f, 771.3602f),
            BodyPart.LEFT_KNEE to PointF(539.98773f, 1107.4801f),
            BodyPart.RIGHT_KNEE to PointF(450.91772f, 1122.4381f),
            BodyPart.LEFT_ANKLE to PointF(493.61127f, 1404.542f),
            BodyPart.RIGHT_ANKLE to PointF(409.42444f, 1488.6501f),
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