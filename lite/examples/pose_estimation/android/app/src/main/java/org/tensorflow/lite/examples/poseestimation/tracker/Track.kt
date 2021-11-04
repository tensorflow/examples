package org.tensorflow.lite.examples.poseestimation.tracker

import org.tensorflow.lite.examples.poseestimation.data.Person

data class Track(
    val person: Person,
    val lastTimestamp: Long
)
