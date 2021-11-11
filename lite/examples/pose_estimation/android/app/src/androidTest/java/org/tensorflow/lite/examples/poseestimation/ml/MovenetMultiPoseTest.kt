/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================
*/

package org.tensorflow.lite.examples.poseestimation.ml

import android.content.Context
import android.graphics.Bitmap
import android.graphics.PointF
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.tensorflow.lite.examples.poseestimation.data.BodyPart
import org.tensorflow.lite.examples.poseestimation.data.Device
import org.tensorflow.lite.examples.poseestimation.ml.MoveNetMultiPose
import org.tensorflow.lite.examples.poseestimation.ml.Type

@RunWith(AndroidJUnit4::class)
class MovenetMultiPoseTest {
    companion object {
        private const val TEST_INPUT_IMAGE1 = "image1.png"
        private const val TEST_INPUT_IMAGE2 = "image2.jpg"
        private const val ACCEPTABLE_ERROR = 17f
    }

    private lateinit var poseDetector: MoveNetMultiPose
    private lateinit var appContext: Context
    private lateinit var inputFinal: Bitmap
    private lateinit var expectedDetectionResult: List<Map<BodyPart, PointF>>

    @Before
    fun setup() {
        appContext = InstrumentationRegistry.getInstrumentation().targetContext
        poseDetector = MoveNetMultiPose.create(appContext, Device.CPU, Type.Dynamic)
        val input1 = EvaluationUtils.loadBitmapAssetByName(TEST_INPUT_IMAGE1)
        val input2 = EvaluationUtils.loadBitmapAssetByName(TEST_INPUT_IMAGE2)
        inputFinal = EvaluationUtils.hConcat(input1, input2)
        expectedDetectionResult =
                EvaluationUtils.loadCSVAsset("pose_landmark_truth.csv")

        // update coordination of the pose_landmark_truth.csv corresponding to the new input image
        for ((_, value) in expectedDetectionResult[1]) {
            value.x = value.x + input1.width
        }
    }

    @Test
    fun testPoseEstimateResult() {
        val persons = poseDetector.estimatePoses(inputFinal)
        assert(persons.size == 2)

        // Sort the results so that the person on the right side come first.
        val sortedPersons = persons.sortedBy { it.boundingBox?.left }

        EvaluationUtils.assertPoseDetectionResult(
                sortedPersons[0],
                expectedDetectionResult[0],
                ACCEPTABLE_ERROR
        )

        EvaluationUtils.assertPoseDetectionResult(
                sortedPersons[1],
                expectedDetectionResult[1],
                ACCEPTABLE_ERROR
        )
    }
}
