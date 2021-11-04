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

import android.graphics.RectF
import androidx.test.ext.junit.runners.AndroidJUnit4
import junit.framework.Assert
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.tensorflow.lite.examples.poseestimation.data.Person

@RunWith(AndroidJUnit4::class)
class BoundingBoxTrackerTest {
    companion object {
        private const val MAX_TRACKS = 4
        private const val MAX_AGE = 1000 // Unit: milliseconds.
        private const val MIN_SIMILARITY = 0.5f
    }

    private lateinit var boundingBoxTracker: BoundingBoxTracker

    @Before
    fun setup() {
        val trackerConfig = TrackerConfig(MAX_TRACKS, MAX_AGE, MIN_SIMILARITY)
        boundingBoxTracker = BoundingBoxTracker(trackerConfig)
    }

    @Test
    fun testIoU() {
        val persons = Person(
            -1, listOf(), RectF(
                0f,
                0f,
                2f / 3,
                1f
            ), 1f
        )

        val track =
            Track(
                Person(
                    -1,
                    listOf(),
                    RectF(
                        1 / 3f,
                        0.0f,
                        1f,
                        1f,
                    ), 1f
                ), 1000000
            )
        val computedIoU = boundingBoxTracker.iou(persons, track.person)
        Assert.assertEquals("Wrong IoU value.", 1f / 3, computedIoU, 0.000001f)
    }

    @Test
    fun testIoUFullOverlap() {
        val persons = Person(
            -1, listOf(),
            RectF(
                0f,
                0f,
                1f,
                1f
            ), 1f
        )

        val track =
            Track(
                Person(
                    -1,
                    listOf(),
                    RectF(
                        0f,
                        0f,
                        1f,
                        1f,
                    ), 1f
                ), 1000000
            )
        val computedIoU = boundingBoxTracker.iou(persons, track.person)
        Assert.assertEquals("Wrong IoU value.", 1f, computedIoU, 0.000001f)
    }

    @Test
    fun testIoUNoIntersection() {
        val persons = Person(
            -1, listOf(),
            RectF(
                0f,
                0f,
                0.5f,
                0.5f
            ), 1f
        )

        val track =
            Track(
                Person(
                    -1,
                    listOf(),
                    RectF(
                        0.5f,
                        0.5f,
                        1f,
                        1f,
                    ), 1f
                ), 1000000
            )
        val computedIoU = boundingBoxTracker.iou(persons, track.person)
        Assert.assertEquals("Wrong IoU value.", 0f, computedIoU, 0.000001f)
    }

    @Test
    fun testBoundingBoxTracking() {
        // Timestamp: 0. Poses becomes the first two tracks.
        var persons = listOf(
            Person( // Becomes track 1.
                -1, listOf(), RectF(
                    0f,
                    0f,
                    0.5f,
                    0.5f,
                ), 1f
            ),
            Person( // Becomes track 2.
                -1, listOf(), RectF(
                    0f,
                    0f,
                    1f,
                    1f
                ), 1f
            )
        )
        persons = boundingBoxTracker.apply(persons, 0)
        var track = boundingBoxTracker.tracks
        Assert.assertEquals(2, persons.size)
        Assert.assertEquals(1, persons[0].id)
        Assert.assertEquals(2, persons[1].id)
        Assert.assertEquals(2, track.size)
        Assert.assertEquals(1, track[0].person.id)
        Assert.assertEquals(0, track[0].lastTimestamp)
        Assert.assertEquals(2, track[1].person.id)
        Assert.assertEquals(0, track[1].lastTimestamp)

        // Timestamp: 100000. First pose is linked with track 1. Second pose spawns
        // a new track (id = 2).
        persons = listOf(
            Person( // Linked with track 1.
                -1, listOf(), RectF(
                    0.1f,
                    0.1f,
                    0.5f,
                    0.5f
                ), 1f
            ),
            Person( // Becomes track 3.
                -1, listOf(), RectF(
                    0.2f,
                    0.3f,
                    0.9f,
                    0.9f
                ), 1f
            )
        )
        persons = boundingBoxTracker.apply(persons, 100000)
        track = boundingBoxTracker.tracks
        Assert.assertEquals(2, persons.size)
        Assert.assertEquals(1, persons[0].id)
        Assert.assertEquals(3, persons[1].id)
        Assert.assertEquals(3, track.size)
        Assert.assertEquals(1, track[0].person.id)
        Assert.assertEquals(100000, track[0].lastTimestamp)
        Assert.assertEquals(3, track[1].person.id)
        Assert.assertEquals(100000, track[1].lastTimestamp)
        Assert.assertEquals(2, track[2].person.id)
        Assert.assertEquals(0, track[2].lastTimestamp)

        // Timestamp: 1050000. First pose is linked with track 1. Second pose is
        // identical to track 2, but is not linked because track 2 is deleted due to
        // age. Instead it spawns track 4.
        persons = listOf(
            Person( // Linked with track 1.
                -1, listOf(), RectF(
                    0.1f,
                    0.1f,
                    0.55f,
                    0.5f
                ), 1f
            ),
            Person( // Becomes track 4.
                -1, listOf(), RectF(
                    0f,
                    0f,
                    1f,
                    1f
                ), 1f
            )
        )
        persons = boundingBoxTracker.apply(persons, 1050000)
        track = boundingBoxTracker.tracks
        Assert.assertEquals(2, persons.size)
        Assert.assertEquals(1, persons[0].id)
        Assert.assertEquals(4, persons[1].id)
        Assert.assertEquals(3, track.size)
        Assert.assertEquals(1, track[0].person.id)
        Assert.assertEquals(1050000, track[0].lastTimestamp)
        Assert.assertEquals(4, track[1].person.id)
        Assert.assertEquals(1050000, track[1].lastTimestamp)
        Assert.assertEquals(3, track[2].person.id)
        Assert.assertEquals(100000, track[2].lastTimestamp)
    }
}