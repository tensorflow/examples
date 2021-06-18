package org.tensorflow.lite.examples.poseestimation

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.google.common.truth.Truth.assertThat
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.tensorflow.lite.examples.poseestimation.data.BodyPart
import org.tensorflow.lite.examples.poseestimation.data.Coordinate
import org.tensorflow.lite.examples.poseestimation.data.Device
import org.tensorflow.lite.examples.poseestimation.data.Person
import org.tensorflow.lite.examples.poseestimation.ml.ModelType
import org.tensorflow.lite.examples.poseestimation.ml.MoveNet
import org.tensorflow.lite.examples.poseestimation.ml.PoseDetector
import org.tensorflow.lite.examples.poseestimation.ml.PoseNet


@RunWith(AndroidJUnit4::class)
class PosenetTest {

    companion object {
        private const val TEST_INPUT_IMAGE1 = "image1"
        private val EXPECTED_DETECTION_RESULT1 = mapOf(
            BodyPart.NOSE to Coordinate(483f,304f),
            BodyPart.LEFT_EYE to Coordinate(501f,283f),
            BodyPart.RIGHT_EYE to Coordinate(477f,282f),
            BodyPart.LEFT_EAR to Coordinate(553f,287f),
            BodyPart.RIGHT_EAR to Coordinate(468f,289f),
            BodyPart.LEFT_SHOULDER to Coordinate(616f,407f),
            BodyPart.RIGHT_SHOULDER to Coordinate(418f,401f),
            BodyPart.LEFT_ELBOW to Coordinate(673f,639f),
            BodyPart.RIGHT_ELBOW to Coordinate(396f,623f),
            BodyPart.LEFT_WRIST to Coordinate(693f,851f),
            BodyPart.RIGHT_WRIST to Coordinate(371f,819f),
            BodyPart.LEFT_HIP to Coordinate(621f,815f),
            BodyPart.RIGHT_HIP to Coordinate(439f,823f),
            BodyPart.LEFT_KNEE to Coordinate(655f,1178f),
            BodyPart.RIGHT_KNEE to Coordinate(366f,1172f),
            BodyPart.LEFT_ANKLE to Coordinate(688f,1381f),
            BodyPart.RIGHT_ANKLE to Coordinate(334f,1386f),
        )

        private const val TEST_INPUT_IMAGE2 = "image2"
        private val EXPECTED_DETECTION_RESULT2 = mapOf(
            BodyPart.NOSE to Coordinate(505f, 265f),
            BodyPart.LEFT_EYE to Coordinate(516f, 253f),
            BodyPart.RIGHT_EYE to Coordinate(498f, 257f),
            BodyPart.LEFT_EAR to Coordinate(524f, 232f),
            BodyPart.RIGHT_EAR to Coordinate(468f, 243f),
            BodyPart.LEFT_SHOULDER to Coordinate(571f, 325f),
            BodyPart.RIGHT_SHOULDER to Coordinate(387f, 351f),
            BodyPart.LEFT_ELBOW to Coordinate(696f, 517f),
            BodyPart.RIGHT_ELBOW to Coordinate(471f, 574f),
            BodyPart.LEFT_WRIST to Coordinate(631f, 566f),
            BodyPart.RIGHT_WRIST to Coordinate(603f, 567f),
            BodyPart.LEFT_HIP to Coordinate(581f, 812f),
            BodyPart.RIGHT_HIP to Coordinate(452f, 757f),
            BodyPart.LEFT_KNEE to Coordinate(560f, 1113f),
            BodyPart.RIGHT_KNEE to Coordinate(459f, 1113f),
            BodyPart.LEFT_ANKLE to Coordinate(482f, 1385f),
            BodyPart.RIGHT_ANKLE to Coordinate(424f, 1414f),
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