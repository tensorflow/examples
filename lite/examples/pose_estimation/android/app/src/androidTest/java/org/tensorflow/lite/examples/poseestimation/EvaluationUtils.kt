package org.tensorflow.lite.examples.poseestimation

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.PointF
import androidx.test.platform.app.InstrumentationRegistry
import com.google.common.truth.Truth.assertThat
import com.google.common.truth.Truth.assertWithMessage
import org.tensorflow.lite.examples.poseestimation.data.BodyPart
import org.tensorflow.lite.examples.poseestimation.data.Person
import kotlin.math.pow

object EvaluationUtils {

    private const val ACCEPTABLE_ERROR = 10f // max 10 pixels
    private const val BITMAP_FIXED_WIDTH_SIZE = 400

    /**
     * Assert whether the detected person from the image match with the expected result.
     * Detection result is accepted as correct if it is within the ACCEPTABLE_ERROR range from the
     * expected result.
     */
    fun assertPoseDetectionResult(
        person: Person,
        expectedResult: Map<BodyPart, PointF>
    ) {
        // Check if the model is confident enough in detecting the person
        assertThat(person.score).isGreaterThan(0.5f)

        for ((bodyPart, expectedPointF) in expectedResult) {
            val keypoint = person.keyPoints.firstOrNull { it.bodyPart == bodyPart }
            assertWithMessage("$bodyPart must exist").that(keypoint).isNotNull()

            val detectedPointF = keypoint!!.coordinate
            val distanceFromExpectedPointF = distance(detectedPointF, expectedPointF)
            assertWithMessage("Detected $bodyPart must be close to expected result")
                .that(distanceFromExpectedPointF).isAtMost(ACCEPTABLE_ERROR)
        }
    }

    /**
     * Load an image from assets folder using its resource name.
     * Note: The image is implicitly resized to a fixed 400px width, while keeping its ratio.
     * This is necessary to keep the test image consistent because different bitmap resolution will
     * be loaded based on the device screen size.
     */
    fun loadBitmapResourceByName(name: String): Bitmap {
        val resources = InstrumentationRegistry.getInstrumentation().context.resources
        val resourceId = resources.getIdentifier(
            name, "drawable",
            InstrumentationRegistry.getInstrumentation().context.packageName
        )
        val options = BitmapFactory.Options()
        options.inMutable = true
        return scaleBitmapToFixedSize(BitmapFactory.decodeResource(resources, resourceId, options))
    }

    private fun scaleBitmapToFixedSize(bitmap: Bitmap): Bitmap {
        val ratio = bitmap.width.toFloat() / bitmap.height
        return Bitmap.createScaledBitmap(
            bitmap,
            BITMAP_FIXED_WIDTH_SIZE,
            (BITMAP_FIXED_WIDTH_SIZE / ratio).toInt(),
            false
        )
    }

    private fun distance(point1: PointF, point2: PointF): Float {
        return ((point1.x - point2.x).pow(2) + (point1.y - point2.y).pow(2)).pow(0.5f)
    }
}