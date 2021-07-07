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

package org.tensorflow.lite.examples.handtracking.handlandmark

import org.tensorflow.lite.examples.handtracking.handlandmark.data.*
import kotlin.math.*

object DetectionHelper {

    /**
     * Specify the rotation angle of the output rect with a vector formed by connecting
     * two keypoints in the detection, together with the target angle (in degrees)
     * of that vector after rotation. The target angle is counter-clockwise starting
     * from the positive x-axis.
     */
    fun computeRotation(
        detection: Detection,
        width: Int,
        height: Int,
        rotationVectorStartKeypointIndex: Int,
        rotationVectorEndKeypointIndex: Int,
        rotationVectorTargetAngleDegrees: Float
    ): Double {
        val targetAngle = Math.PI * rotationVectorTargetAngleDegrees / 180f
        val locationData = detection.locationData
        val x0 =
            locationData.relativeKeyPoints[rotationVectorStartKeypointIndex].x * width
        val y0 =
            locationData.relativeKeyPoints[rotationVectorStartKeypointIndex].y * height
        val x1 = locationData.relativeKeyPoints[rotationVectorEndKeypointIndex].x * width
        val y1 = locationData.relativeKeyPoints[rotationVectorEndKeypointIndex].y * height

        // return rotation
        return normalizeRadians(targetAngle - atan2(-(y1 - y0), x1 - x0))
    }

    /**
     * Apply specified transformation on the input rectangle and return it.
     */
    fun rectTransformation(
        normalizedRect: Rect,
        imageWidth: Int,
        imageHeight: Int,
        scaleX: Float,
        scaleY: Float,
        shiftX: Float,
        shiftY: Float
    ): Rect {
        var width = normalizedRect.width
        var height = normalizedRect.height
        val rotation = normalizedRect.rotation

        if (rotation == 0f) {
            normalizedRect.centerX = normalizedRect.centerX + width * shiftX
            normalizedRect.centerY = normalizedRect.centerY + height * shiftY
        } else {
            val xShift =
                (imageWidth * width * shiftX * cos(rotation) - imageHeight * height * shiftY * sin(
                    rotation
                )) / imageWidth
            val yShift =
                (imageWidth * width * shiftX * sin(rotation) + imageHeight * height * shiftY * cos(
                    rotation
                )) / imageHeight
            normalizedRect.centerX = normalizedRect.centerX + xShift
            normalizedRect.centerY = normalizedRect.centerY + yShift
        }

        val longSide = max(width * imageWidth, height * imageHeight)
        width = longSide / imageWidth
        height = longSide / imageHeight

        normalizedRect.width = (width * scaleX)
        normalizedRect.height = (height * scaleY)
        return normalizedRect
    }

    /**
     * Pads ROI(region of interest), so extraction happens correctly if aspect ratio is to be kept.
     * Returns letterbox padding applied.
     */
    fun padRoi(
        inputTensorWidth: Int,
        inputTensorHeight: Int,
        keepAspectRatio: Boolean,
        roi: Rect
    ): FloatArray {
        val padRoi = FloatArray(4) { 0f }
        if (!keepAspectRatio) return padRoi

        val tensorAspectRatio = inputTensorHeight.toFloat() / inputTensorWidth.toFloat()
        val roiAspectRatio = roi.height / roi.width
        var verticalPadding = 0f
        var horizonPadding = 0f
        val newWidth: Float
        val newHeight: Float
        if (tensorAspectRatio > roiAspectRatio) {
            newWidth = roi.width
            newHeight = roi.width * tensorAspectRatio
            verticalPadding = (1f - roiAspectRatio / tensorAspectRatio) / 2f
        } else {
            newWidth = roi.height / tensorAspectRatio
            newHeight = roi.height
            horizonPadding = (1f - tensorAspectRatio / roiAspectRatio) / 2f
        }
        roi.width = newWidth
        roi.height = newHeight
        padRoi[0] = horizonPadding
        padRoi[1] = verticalPadding
        padRoi[2] = horizonPadding
        padRoi[3] = verticalPadding
        return padRoi
    }

    /**
     * Return Letterbox padding from the 4 sides ([left, top, right, bottom]) of output image.
     * normalized to [.0f, 1.f]
     */
    fun getLetterBoxPadding(
        inputWidth: Int,
        inputHeight: Int,
        outputTensorWidth: Int,
        outputTensorHeight: Int,
        keepAspectRatio: Boolean = true
    ): FloatArray {
        val roi = getRoi(inputWidth, inputHeight)
        return padRoi(outputTensorWidth, outputTensorHeight, keepAspectRatio, roi)
    }

    /**
     * Generate new ROI(region of interest) or converts it from normalized rect.
     */
    private fun getRoi(inputWidth: Int, inputHeight: Int): Rect {
        return Rect(
            0.5f * inputWidth,
            0.5f * inputHeight,
            inputWidth.toFloat(),
            inputHeight.toFloat()
        )
    }

    /**
     * Convert relative box bounding to normalize rect
     */
    fun rectFromBox(box: RelativeBoundingBox): Rect {
        return Rect(
            box.xMin + box.width / 2f,
            box.yMin + box.height / 2f,
            box.width,
            box.height
        )
    }

    private fun normalizeRadians(angle: Double): Double {
        return angle - 2f * Math.PI * floor((angle - (-Math.PI)) / (2f * Math.PI))
    }
}
