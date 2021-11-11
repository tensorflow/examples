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

import android.graphics.PointF
import androidx.test.ext.junit.runners.AndroidJUnit4
import junit.framework.Assert
import junit.framework.TestCase.assertEquals
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.tensorflow.lite.examples.poseestimation.data.BodyPart
import org.tensorflow.lite.examples.poseestimation.data.KeyPoint
import org.tensorflow.lite.examples.poseestimation.data.Person
import kotlin.math.exp
import kotlin.math.pow

@RunWith(AndroidJUnit4::class)
class KeyPointsTrackerTest {
    companion object {
        private const val MAX_TRACKS = 4
        private const val MAX_AGE = 1000
        private const val MIN_SIMILARITY = 0.5f
        private const val KEYPOINT_THRESHOLD = 0.2f
        private const val MIN_NUM_KEYPOINT = 2
        private val KEYPOINT_FALLOFF = listOf(0.1f, 0.1f, 0.1f, 0.1f)
    }

    private lateinit var keyPointsTracker: KeyPointsTracker

    @Before
    fun setup() {
        val trackerConfig = TrackerConfig(
            MAX_TRACKS, MAX_AGE, MIN_SIMILARITY,
            KeyPointsTrackerParams(KEYPOINT_THRESHOLD, KEYPOINT_FALLOFF, MIN_NUM_KEYPOINT)
        )
        keyPointsTracker = KeyPointsTracker(trackerConfig)
    }

    @Test
    fun testOks() {
        val persons =
            Person(
                -1, listOf(
                    KeyPoint(BodyPart.NOSE, PointF(0.2f, 0.2f), 1f),
                    KeyPoint(BodyPart.RIGHT_ELBOW, PointF(0.4f, 0.4f), 0.8f),
                    KeyPoint(BodyPart.RIGHT_KNEE, PointF(0.6f, 0.6f), 0.1f),
                    KeyPoint(BodyPart.RIGHT_ANKLE, PointF(0.8f, 0.7f), 0.8f),
                ), score = 1f
            )
        val tracks =
            Track(
                Person(
                    0, listOf(
                        KeyPoint(BodyPart.NOSE, PointF(0.2f, 0.2f), 1f),
                        KeyPoint(BodyPart.RIGHT_ELBOW, PointF(0.4f, 0.4f), 0.8f),
                        KeyPoint(BodyPart.RIGHT_KNEE, PointF(0.6f, 0.6f), 0.9f),
                        KeyPoint(BodyPart.RIGHT_ANKLE, PointF(0.8f, 0.8f), 0.8f),
                    ), score = 1f
                ),
                1000000,
            )

        val oks = keyPointsTracker.oks(persons, tracks.person)
        val boxArea = (0.8f - 0.2f) * (0.8f - 0.2f)
        val x = 2f * KEYPOINT_FALLOFF[3]
        val d = 0.1f
        val expectedOks: Float =
            (1f + 1f + exp(-1f * d.pow(2) / (2f * boxArea * x.pow(2)))) / 3f
        assertEquals(expectedOks, oks, 0.000001f)
    }

    @Test
    fun testOksReturnZero() {
        // Compute OKS returns 0.0 with less than 2 valid keypoints
        val persons =
            Person(
                -1, listOf(
                    KeyPoint(BodyPart.NOSE, PointF(0.2f, 0.2f), 1f),
                    KeyPoint(BodyPart.RIGHT_ELBOW, PointF(0.4f, 0.4f), 0.1f), // Low confidence.
                    KeyPoint(BodyPart.RIGHT_KNEE, PointF(0.6f, 0.6f), 0.9f),
                    KeyPoint(BodyPart.RIGHT_ANKLE, PointF(0.8f, 0.8f), 0.8f),
                ), score = 1f
            )
        val tracks =
            Track(
                Person(
                    0, listOf(
                        KeyPoint(BodyPart.NOSE, PointF(0.2f, 0.2f), 1f),
                        KeyPoint(BodyPart.RIGHT_ELBOW, PointF(0.4f, 0.4f), 0.8f),
                        KeyPoint(BodyPart.RIGHT_KNEE, PointF(0.6f, 0.6f), 0.1f),// Low confidence.
                        KeyPoint(BodyPart.RIGHT_ANKLE, PointF(0.8f, 0.8f), 0.0f),// Low confidence.
                    ), score = 1f
                ), 1000000
            )

        val oks = keyPointsTracker.oks(persons, tracks.person)
        assertEquals(0f, oks, 0.000001f)
    }

    @Test
    fun testArea() {
        val keyPoints = listOf(
            KeyPoint(BodyPart.NOSE, PointF(0.1f, 0.2f), 1f),
            KeyPoint(BodyPart.RIGHT_ELBOW, PointF(0.3f, 0.4f), 0.9f),
            KeyPoint(BodyPart.RIGHT_KNEE, PointF(0.4f, 0.6f), 0.9f),
            KeyPoint(BodyPart.RIGHT_ANKLE, PointF(0.7f, 0.8f), 0.1f),
        )
        val area = keyPointsTracker.area(keyPoints)
        val expectedArea = (0.4f - 0.1f) * (0.6f - 0.2f)
        assertEquals(expectedArea, area)
    }

    @Test
    fun testKeyPointsTracker() {
        // Timestamp: 0. Person becomes the only track.
        var persons = listOf(
            Person(
                -1, listOf(
                    KeyPoint(BodyPart.NOSE, PointF(0.2f, 0.2f), 1f),
                    KeyPoint(BodyPart.RIGHT_ELBOW, PointF(0.4f, 0.4f), 0.8f),
                    KeyPoint(BodyPart.RIGHT_KNEE, PointF(0.6f, 0.6f), 0.9f),
                    KeyPoint(BodyPart.RIGHT_ANKLE, PointF(0.8f, 0.8f), 0.0f),
                ), score = 0.9f
            )
        )
        persons = keyPointsTracker.apply(persons, 0)
        var track = keyPointsTracker.tracks
        assertEquals(1, persons.size)
        assertEquals(1, persons[0].id)
        assertEquals(1, track.size)
        assertEquals(1, track[0].person.id)
        assertEquals(0, track[0].lastTimestamp)

        // Timestamp: 100000. First person is linked with track 1. Second person spawns
        // a new track (id = 2).
        persons = listOf(
            Person(
                -1,
                listOf(
                    // Links with id = 1.
                    KeyPoint(BodyPart.NOSE, PointF(0.2f, 0.2f), 1f),
                    KeyPoint(BodyPart.RIGHT_ELBOW, PointF(0.4f, 0.4f), 0.8f),
                    KeyPoint(BodyPart.RIGHT_KNEE, PointF(0.6f, 0.6f), 0.9f),
                    KeyPoint(BodyPart.RIGHT_ANKLE, PointF(0.8f, 0.8f), 0.8f),
                ),
                score = 1f
            ),
            Person(
                -1,
                listOf(
                    // Becomes id = 2.
                    KeyPoint(BodyPart.NOSE, PointF(0.8f, 0.8f), 0.8f),
                    KeyPoint(BodyPart.RIGHT_ELBOW, PointF(0.6f, 0.6f), 0.3f),
                    KeyPoint(BodyPart.RIGHT_KNEE, PointF(0.4f, 0.4f), 0.1f),
                    KeyPoint(BodyPart.RIGHT_ANKLE, PointF(0.2f, 0.2f), 0.8f),
                ),
                score = 1f
            )
        )
        persons = keyPointsTracker.apply(persons, 100000)
        track = keyPointsTracker.tracks
        assertEquals(2, persons.size)
        assertEquals(1, persons[0].id)
        assertEquals(2, persons[1].id)
        assertEquals(2, track.size)
        assertEquals(1, track[0].person.id)
        assertEquals(100000, track[0].lastTimestamp)
        assertEquals(2, track[1].person.id)
        assertEquals(100000, track[1].lastTimestamp)

        // Timestamp: 900000. First person is linked with track 2. Second person spawns
        // a new track (id = 3).
        persons = listOf(
            Person(
                -1,
                listOf(
                    // Links with id = 2.
                    KeyPoint(BodyPart.NOSE, PointF(0.6f, 0.7f), 0.7f),
                    KeyPoint(BodyPart.RIGHT_ELBOW, PointF(0.5f, 0.6f), 0.7f),
                    KeyPoint(BodyPart.RIGHT_KNEE, PointF(0.0f, 0.0f), 0.1f),
                    KeyPoint(BodyPart.RIGHT_ANKLE, PointF(0.2f, 0.1f), 1f),
                ),
                score = 1f
            ),
            Person(
                -1,
                listOf(
                    // Becomes id = 3.
                    KeyPoint(BodyPart.NOSE, PointF(0.5f, 0.1f), 0.6f),
                    KeyPoint(BodyPart.RIGHT_ELBOW, PointF(0.9f, 0.3f), 0.6f),
                    KeyPoint(BodyPart.RIGHT_KNEE, PointF(0.1f, 1f), 0.9f),
                    KeyPoint(BodyPart.RIGHT_ANKLE, PointF(0.4f, 0.4f), 0.1f),
                ),
                score = 1f
            )
        )
        persons = keyPointsTracker.apply(persons, 900000)
        track = keyPointsTracker.tracks
        assertEquals(2, persons.size)
        assertEquals(2, persons[0].id)
        assertEquals(3, persons[1].id)
        assertEquals(3, track.size)
        assertEquals(2, track[0].person.id)
        assertEquals(900000, track[0].lastTimestamp)
        assertEquals(3, track[1].person.id)
        assertEquals(900000, track[1].lastTimestamp)
        assertEquals(1, track[2].person.id)
        assertEquals(100000, track[2].lastTimestamp)

        // Timestamp: 1200000. First person spawns a new track (id = 4), even though
        // it has the same keypoints as track 1. This is because the age exceeds
        // 1000 msec. The second person links with id 2. The third person spawns a new
        // track (id = 5).
        persons = listOf(
            Person(
                -1,
                listOf(
                    // Becomes id = 4.
                    KeyPoint(BodyPart.NOSE, PointF(0.2f, 0.2f), 1f),
                    KeyPoint(BodyPart.RIGHT_ELBOW, PointF(0.4f, 0.4f), 0.8f),
                    KeyPoint(BodyPart.RIGHT_KNEE, PointF(0.6f, 0.6f), 0.9f),
                    KeyPoint(BodyPart.RIGHT_ANKLE, PointF(0.8f, 0.8f), 0.8f),
                ),
                score = 1f
            ),
            Person(
                -1,
                listOf(
                    // Links with id = 2.
                    KeyPoint(BodyPart.NOSE, PointF(0.55f, 0.7f), 0.7f),
                    KeyPoint(BodyPart.RIGHT_ELBOW, PointF(0.5f, 0.6f), 0.9f),
                    KeyPoint(BodyPart.RIGHT_KNEE, PointF(1f, 1f), 0.1f),
                    KeyPoint(BodyPart.RIGHT_ANKLE, PointF(0.8f, 0.1f), 0f),
                ),
                score = 1f
            ),
            Person(
                -1,
                listOf(
                    // Becomes id = 5.
                    KeyPoint(BodyPart.NOSE, PointF(0.1f, 0.1f), 0.1f),
                    KeyPoint(BodyPart.RIGHT_ELBOW, PointF(0.2f, 0.2f), 0.9f),
                    KeyPoint(BodyPart.RIGHT_KNEE, PointF(0.3f, 0.3f), 0.7f),
                    KeyPoint(BodyPart.RIGHT_ANKLE, PointF(0.4f, 0.4f), 0.8f),
                ),
                score = 1f
            )
        )
        persons = keyPointsTracker.apply(persons, 1200000)
        track = keyPointsTracker.tracks
        assertEquals(3, persons.size)
        assertEquals(4, persons[0].id)
        assertEquals(2, persons[1].id)
        assertEquals(4, track.size)
        assertEquals(2, track[0].person.id)
        assertEquals(1200000, track[0].lastTimestamp)
        assertEquals(4, track[1].person.id)
        assertEquals(1200000, track[1].lastTimestamp)
        assertEquals(5, track[2].person.id)
        assertEquals(1200000, track[2].lastTimestamp)
        assertEquals(3, track[3].person.id)
        assertEquals(900000, track[3].lastTimestamp)

        // Timestamp: 1300000. First person spawns a new track (id = 6). Since
        // maxTracks is 4, the oldest track (id = 3) is removed.
        persons = listOf(
            Person(
                -1,
                listOf(
                    // Becomes id = 6.
                    KeyPoint(BodyPart.NOSE, PointF(0.1f, 0.8f), 1f),
                    KeyPoint(BodyPart.RIGHT_ELBOW, PointF(0.2f, 0.9f), 0.6f),
                    KeyPoint(BodyPart.RIGHT_KNEE, PointF(0.2f, 0.9f), 0.5f),
                    KeyPoint(BodyPart.RIGHT_ANKLE, PointF(0.8f, 0.2f), 0.4f),
                ),
                score = 1f
            )
        )
        persons = keyPointsTracker.apply(persons, 1300000)
        track = keyPointsTracker.tracks
        assertEquals(1, persons.size)
        assertEquals(6, persons[0].id)
        assertEquals(4, track.size)
        assertEquals(6, track[0].person.id)
        assertEquals(1300000, track[0].lastTimestamp)
        assertEquals(2, track[1].person.id)
        assertEquals(1200000, track[1].lastTimestamp)
        assertEquals(4, track[2].person.id)
        assertEquals(1200000, track[2].lastTimestamp)
        assertEquals(5, track[3].person.id)
        assertEquals(1200000, track[3].lastTimestamp)
    }
}