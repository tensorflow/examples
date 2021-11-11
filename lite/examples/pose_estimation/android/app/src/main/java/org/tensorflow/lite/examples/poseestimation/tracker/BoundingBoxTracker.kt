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

package org.tensorflow.lite.examples.poseestimation.tracker

import androidx.annotation.VisibleForTesting
import org.tensorflow.lite.examples.poseestimation.data.Person
import kotlin.math.max
import kotlin.math.min

/**
 * BoundingBoxTracker, which tracks objects based on bounding box similarity,
 * currently defined as intersection-over-union (IoU).
 */
class BoundingBoxTracker(config: TrackerConfig = TrackerConfig()) : AbstractTracker(config) {

    /**
     * Computes similarity based on intersection-over-union (IoU). See `AbstractTracker`
     * for more details.
     */
    override fun computeSimilarity(persons: List<Person>): List<List<Float>> {
        if (persons.isEmpty() && tracks.isEmpty()) {
            return emptyList()
        }
        return persons.map { person -> tracks.map { track -> iou(person, track.person) } }
    }

    /**
     * Computes the intersection-over-union (IoU) between a person and a track person.
     * @param person1 A person
     * @param person2 A track person
     * @return The IoU  between the person and the track person. This number is
     * between 0 and 1, and larger values indicate more box similarity.
     */
    @VisibleForTesting(otherwise = VisibleForTesting.PRIVATE)
    fun iou(person1: Person, person2: Person): Float {
        if (person1.boundingBox != null && person2.boundingBox != null) {
            val xMin = max(person1.boundingBox.left, person2.boundingBox.left)
            val yMin = max(person1.boundingBox.top, person2.boundingBox.top)
            val xMax = min(person1.boundingBox.right, person2.boundingBox.right)
            val yMax = min(person1.boundingBox.bottom, person2.boundingBox.bottom)
            if (xMin >= xMax || yMin >= yMax) return 0f
            val intersection = (xMax - xMin) * (yMax - yMin)
            val areaPerson = person1.boundingBox.width() * person1.boundingBox.height()
            val areaTrack = person2.boundingBox.width() * person2.boundingBox.height()
            return intersection / (areaPerson + areaTrack - intersection)
        }
        return 0f
    }
}
