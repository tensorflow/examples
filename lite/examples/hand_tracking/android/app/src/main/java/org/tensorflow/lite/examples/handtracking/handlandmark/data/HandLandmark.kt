//Copyright 2021 Google LLC
//
//Licensed under the Apache License, Version 2.0 (the "License");
//you may not use this file except in compliance with the License.
//You may obtain a copy of the License at
//
//https://www.apache.org/licenses/LICENSE-2.0
//
//Unless required by applicable law or agreed to in writing, software
//distributed under the License is distributed on an "AS IS" BASIS,
//WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//See the License for the specific language governing permissions and
//limitations under the License.

package org.tensorflow.lite.examples.handtracking.handlandmark.data

/**
 * An entity representing a hand landmark location and type
 */
data class HandLandmark(
    val type: Int,
    val x: Float,
    val y: Float,
    val z: Float
) {

    companion object {
        // Values representing the hand landmark types that the model can detect
        const val WRIST = 0
        const val THUMB_CMC = 1
        const val THUMB_MCP = 2
        const val THUMB_IP = 3
        const val THUMB_TIP = 4
        const val INDEX_FINGER_MCP = 5
        const val INDEX_FINGER_PIP = 6
        const val INDEX_FINGER_DIP = 7
        const val INDEX_FINGER_TIP = 8
        const val MIDDLE_FINGER_MCP = 9
        const val MIDDLE_FINGER_PIP = 10
        const val MIDDLE_FINGER_DIP = 11
        const val MIDDLE_FINGER_TIP = 12
        const val RING_FINGER_MCP = 13
        const val RING_FINGER_PIP = 14
        const val RING_FINGER_DIP = 15
        const val RING_FINGER_TIP = 16
        const val PINKY_MCP = 17
        const val PINKY_PIP = 18
        const val PINKY_DIP = 19
        const val PINKY_TIP = 20
    }
}

