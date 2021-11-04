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
import android.graphics.PointF
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.tensorflow.lite.examples.poseestimation.data.BodyPart
import org.tensorflow.lite.examples.poseestimation.data.Device

@RunWith(AndroidJUnit4::class)
class MovenetThunderTest {

    companion object {
        private const val TEST_INPUT_IMAGE1 = "image1.png"
        private const val TEST_INPUT_IMAGE2 = "image2.jpg"
        private const val ACCEPTABLE_ERROR = 15f
    }

    private lateinit var poseDetector: PoseDetector
    private lateinit var appContext: Context
    private lateinit var expectedDetectionResult: List<Map<BodyPart, PointF>>

    @Before
    fun setup() {
        appContext = InstrumentationRegistry.getInstrumentation().targetContext
        poseDetector = MoveNet.create(appContext, Device.CPU, ModelType.Thunder)
        expectedDetectionResult =
            EvaluationUtils.loadCSVAsset("pose_landmark_truth.csv")
    }

    @Test
    fun testPoseEstimationResultWithImage1() {
        val input = EvaluationUtils.loadBitmapAssetByName(TEST_INPUT_IMAGE1)

        // As Movenet use previous frame to optimize detection result, we run it multiple times
        // using the same image to improve result.
        poseDetector.estimatePoses(input)
        poseDetector.estimatePoses(input)
        poseDetector.estimatePoses(input)
        val person = poseDetector.estimatePoses(input)[0]
        EvaluationUtils.assertPoseDetectionResult(
            person,
            expectedDetectionResult[0],
            ACCEPTABLE_ERROR
        )
    }

    @Test
    fun testPoseEstimationResultWithImage2() {
        val input = EvaluationUtils.loadBitmapAssetByName(TEST_INPUT_IMAGE2)

        // As Movenet use previous frame to optimize detection result, we run it multiple times
        // using the same image to improve result.
        poseDetector.estimatePoses(input)
        poseDetector.estimatePoses(input)
        poseDetector.estimatePoses(input)
        val person = poseDetector.estimatePoses(input)[0]
        EvaluationUtils.assertPoseDetectionResult(
            person,
            expectedDetectionResult[1],
            ACCEPTABLE_ERROR
        )
    }
}