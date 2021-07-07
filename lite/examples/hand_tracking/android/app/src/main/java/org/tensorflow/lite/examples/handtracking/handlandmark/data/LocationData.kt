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

import android.graphics.RectF

data class LocationData(
    var relativeBoundingBox: RelativeBoundingBox = RelativeBoundingBox(0f,0f,0f,0f),
    var relativeKeyPoints: MutableList<RelativeKeyPoints> = mutableListOf()
) {
    fun getRelativeBoundingBox(): RectF {
        return RectF(
            relativeBoundingBox.xMin,
            relativeBoundingBox.yMin,
            relativeBoundingBox.width,
            relativeBoundingBox.height
        )
    }
}

data class RelativeBoundingBox(
    var xMin: Float,
    var yMin: Float,
    var width: Float,
    var height: Float
)

data class RelativeKeyPoints(
    var x: Float = 0f,
    var y: Float = 0f,
    val score: Float = 0f
)
