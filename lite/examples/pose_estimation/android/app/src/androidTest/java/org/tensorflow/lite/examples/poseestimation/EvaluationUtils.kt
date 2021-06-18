package org.tensorflow.lite.examples.poseestimation

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.test.platform.app.InstrumentationRegistry
import com.google.common.truth.Truth
import com.google.common.truth.Truth.assertThat
import com.google.common.truth.Truth.assertWithMessage
import org.tensorflow.lite.examples.poseestimation.data.BodyPart
import org.tensorflow.lite.examples.poseestimation.data.Coordinate
import org.tensorflow.lite.examples.poseestimation.data.Person
import kotlin.math.pow

object EvaluationUtils {

    private const val ACCEPTABLE_ERROR = 3f // max 3 pixels

    private fun distance(point1: Coordinate, point2: Coordinate) : Float {
        return ((point1.x - point2.x).pow(2) + (point1.y - point2.y).pow(2)).pow(0.5f)
    }

    fun assertPoseDetectionResult(
        person: Person,
        expectedResult: Map<BodyPart, Coordinate>
    ) {
        // Check if the model is confident enough in detecting the person
        assertThat(person.score).isGreaterThan(0.5f)

        for ((bodyPart, expectedCoordinate) in expectedResult) {
            val keypoint = person.keyPoints.firstOrNull { it.bodyPart == bodyPart }
            assertWithMessage("$bodyPart must exist").that(keypoint).isNotNull()

            val detectedCoordinate = keypoint!!.coordinate
            val distanceFromExpectedCoordinate = distance(detectedCoordinate, expectedCoordinate)
            assertWithMessage("Detected $bodyPart must be close to expected result")
                .that(distanceFromExpectedCoordinate).isAtMost(ACCEPTABLE_ERROR)
        }
    }

    fun loadBitmapResourceByName(name: String): Bitmap {
        val resources = InstrumentationRegistry.getInstrumentation().context.resources
        val resourceId = resources.getIdentifier(
            name, "drawable",
            InstrumentationRegistry.getInstrumentation().context.packageName
        )
        val options = BitmapFactory.Options()
        options.inMutable = true
        return BitmapFactory.decodeResource(resources, resourceId, options)
    }

}