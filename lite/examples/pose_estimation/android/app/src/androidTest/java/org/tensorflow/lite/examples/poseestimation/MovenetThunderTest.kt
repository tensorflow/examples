package org.tensorflow.lite.examples.poseestimation

import android.content.Context
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.tensorflow.lite.examples.poseestimation.data.BodyPart
import org.tensorflow.lite.examples.poseestimation.data.Coordinate
import org.tensorflow.lite.examples.poseestimation.data.Device
import org.tensorflow.lite.examples.poseestimation.ml.ModelType
import org.tensorflow.lite.examples.poseestimation.ml.MoveNet
import org.tensorflow.lite.examples.poseestimation.ml.PoseDetector


@RunWith(AndroidJUnit4::class)
class MovenetThunderTest {

    companion object {

        private const val TEST_INPUT_IMAGE1 = "image1"
        private val EXPECTED_DETECTION_RESULT1 = mapOf(
            BodyPart.NOSE to Coordinate(478.6269f,229.22507f),
            BodyPart.LEFT_EYE to Coordinate(522.0537f,198.03152f),
            BodyPart.RIGHT_EYE to Coordinate(460.26233f,200.85661f),
            BodyPart.LEFT_EAR to Coordinate(593.27924f,230.2767f),
            BodyPart.RIGHT_EAR to Coordinate(459.25964f,218.3077f),
            BodyPart.LEFT_SHOULDER to Coordinate(637.6053f,396.35214f),
            BodyPart.RIGHT_SHOULDER to Coordinate(383.64044f,376.68973f),
            BodyPart.LEFT_ELBOW to Coordinate(676.7501f,649.512f),
            BodyPart.RIGHT_ELBOW to Coordinate(385.07837f,618.25934f),
            BodyPart.LEFT_WRIST to Coordinate(694.23096f,859.5168f),
            BodyPart.RIGHT_WRIST to Coordinate(365.3327f,814.73865f),
            BodyPart.LEFT_HIP to Coordinate(611.20874f,826.57465f),
            BodyPart.RIGHT_HIP to Coordinate(456.81763f,825.30865f),
            BodyPart.LEFT_KNEE to Coordinate(656.32043f,1175.9138f),
            BodyPart.RIGHT_KNEE to Coordinate(360.6452f,1181.5221f),
            BodyPart.LEFT_ANKLE to Coordinate(687.82117f,1480.2277f),
            BodyPart.RIGHT_ANKLE to Coordinate(241.09613f,1492.2631f),
        )

        private const val TEST_INPUT_IMAGE2 = "image2"
        private val EXPECTED_DETECTION_RESULT2 = mapOf(
            BodyPart.NOSE to Coordinate(475.6156f, 223.3255f),
            BodyPart.LEFT_EYE to Coordinate(501.60004f, 191.89624f),
            BodyPart.RIGHT_EYE to Coordinate(447.2423f, 194.20554f),
            BodyPart.LEFT_EAR to Coordinate(532.16705f, 197.10046f),
            BodyPart.RIGHT_EAR to Coordinate(413.48755f, 205.88823f),
            BodyPart.LEFT_SHOULDER to Coordinate(585.81744f, 312.79794f),
            BodyPart.RIGHT_SHOULDER to Coordinate(367.34985f, 359.62976f),
            BodyPart.LEFT_ELBOW to Coordinate(693.0083f, 512.0245f),
            BodyPart.RIGHT_ELBOW to Coordinate(469.50916f, 589.2954f),
            BodyPart.LEFT_WRIST to Coordinate(669.02014f, 579.0159f),
            BodyPart.RIGHT_WRIST to Coordinate(616.79706f, 588.3988f),
            BodyPart.LEFT_HIP to Coordinate(595.7611f, 754.31604f),
            BodyPart.RIGHT_HIP to Coordinate(465.05035f, 762.88824f),
            BodyPart.LEFT_KNEE to Coordinate(540.52106f, 1104.6038f),
            BodyPart.RIGHT_KNEE to Coordinate(448.4724f, 1109.5964f),
            BodyPart.LEFT_ANKLE to Coordinate(476.00775f, 1428.587f),
            BodyPart.RIGHT_ANKLE to Coordinate(396.21613f, 1485.1898f),
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