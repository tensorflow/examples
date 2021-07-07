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

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.examples.handtracking.handlandmark.data.Detection
import org.tensorflow.lite.examples.handtracking.handlandmark.data.HandLandmark
import org.tensorflow.lite.examples.handtracking.handlandmark.data.Rect
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.lang.RuntimeException
import kotlin.math.cos
import kotlin.math.sin

class HandDetector(private val interpreter: Interpreter, private val palmDetector: PalmDetector) {

    companion object {
        private const val MODEL_PATH = "hand_landmark.tflite"
        private val handLandmarkShape = intArrayOf(1, 63)
        private val handFlagShape = intArrayOf(1, 1)
        private const val IMAGE_MEAN = 0f
        private const val IMAGE_STD = 255f
        private const val INPUT_WIDTH = 224
        private const val INPUT_HEIGHT = 224
        private const val NUM_LANDMARK = 21

        fun create(context: Context): HandDetector {
            // Initialize an interpreter with GPU but fallback to CPU if initialization failed.
            val interpreter = try {
                Interpreter(
                    FileUtil.loadMappedFile(context, MODEL_PATH),
                    Interpreter.Options().apply {
                        addDelegate(GpuDelegate())
                    })
            } catch (e: RuntimeException) {
                Interpreter(
                    FileUtil.loadMappedFile(context, MODEL_PATH), null)
            }
            return HandDetector(interpreter, PalmDetector.create(context))
        }
    }

    /**
     * First, detect palms. If no palm detected, return an empty list.
     * Second, use the detected palm information and proceed to detect hand landmarks.
     */
    fun process(inputImage: Bitmap): List<HandLandmark> {
        // If no palm detected, return an empty list of landmarks
        palmDetector.process(inputImage)?.let {
            return landmarkDetection(inputImage, it)
        }
        return emptyList()
    }

    // Releases resources if no longer used.
    fun close() {
        palmDetector.close()
        interpreter.close()
    }

    // Execute hand landmark detection
    private fun landmarkDetection(inputImage: Bitmap, detection: Detection): List<HandLandmark> {
        DetectionHelper.rectFromBox(detection.locationData.relativeBoundingBox)
            .let { normalizedRect ->

                //Rotate the image so that the line connecting center of the wrist and MCP of the
                // middle finger aligns with the Y-axis of the rectangle.
                normalizedRect.rotation = DetectionHelper.computeRotation(
                    detection,
                    inputImage.width,
                    inputImage.height,
                    0,
                    2,
                    90f
                ).toFloat()
                // Expands and shifts the rectangle that contains
                // the palm so that it's likely to cover the entire hand.
                DetectionHelper.rectTransformation(
                    normalizedRect,
                    inputImage.width,
                    inputImage.height,
                    2.6f,
                    2.6f,
                    shiftX = 0f,
                    shiftY = -0.5f
                ).let { normalizedHandRect ->
                    // rotate, resize and translate input image before detect
                    val roiWidth = normalizedHandRect.width * inputImage.width
                    val roiHeight = normalizedHandRect.height * inputImage.height
                    val xCenter = normalizedHandRect.centerX * inputImage.width
                    val yCenter = normalizedHandRect.centerY * inputImage.height
                    val degree = (normalizedHandRect.rotation * 180f / Math.PI).toFloat()
                    val src = floatArrayOf(
                        0f,
                        0f,
                        roiWidth,
                        0f,
                        roiWidth,
                        roiHeight,
                        0f,
                        roiHeight,
                        xCenter,
                        yCenter
                    )
                    val dst = floatArrayOf(
                        0f,
                        0f,
                        INPUT_WIDTH.toFloat(),
                        0f,
                        INPUT_WIDTH.toFloat(),
                        INPUT_HEIGHT.toFloat(),
                        0f,
                        INPUT_HEIGHT.toFloat()
                    )
                    val matrix = Matrix()
                    matrix.postRotate(degree, xCenter, yCenter)
                    matrix.mapPoints(src)
                    matrix.reset()
                    matrix.setPolyToPoly(src, 0, dst, 0, 4)
                    matrix.mapPoints(src)
                    val translateX = INPUT_WIDTH / 2f - src[8]
                    val translateY = INPUT_HEIGHT / 2f - src[9]
                    matrix.postTranslate(translateX, translateY)
                    val detectionBitmap =
                        Bitmap.createBitmap(
                            INPUT_WIDTH,
                            INPUT_HEIGHT, Bitmap.Config.ARGB_8888
                        )
                    val canvas = Canvas(detectionBitmap)
                    canvas.drawBitmap(inputImage, matrix, null)

                    val imageProcessor = ImageProcessor.Builder()
                        .add(
                            NormalizeOp(
                                IMAGE_MEAN,
                                IMAGE_STD
                            )
                        )
                        .build()

                    // Creates inputs for reference.
                    val inputTensorImage =
                        imageProcessor.process(TensorImage(DataType.FLOAT32).apply {
                            load(detectionBitmap)
                        })
                    val handLandmarkTensor =
                        TensorBuffer.createFixedSize(handLandmarkShape, DataType.FLOAT32)
                    val handFlagTensor =
                        TensorBuffer.createFixedSize(handFlagShape, DataType.FLOAT32)
                    val outputs = mapOf(0 to handLandmarkTensor.buffer, 1 to handFlagTensor.buffer)

                    // Runs model inference and gets result.
                    interpreter.runForMultipleInputsOutputs(
                        arrayOf(inputTensorImage.tensorBuffer.buffer),
                        outputs
                    )
                    val handPresenceScore = handFlagTensor.floatArray[0]
                    if (handPresenceScore >= 0.5f) {
                        val landmark = tensorsToLandmarksCalculator(handLandmarkTensor)
                        val letterBoxPadding = DetectionHelper.padRoi(
                            INPUT_WIDTH,
                            INPUT_HEIGHT,
                            true,
                            convertNormalizedRectToRect(
                                inputImage.width,
                                inputImage.height,
                                normalizedHandRect
                            )
                        )
                        val scaleLandmark = landmarkLetterBoxRemoval(landmark, letterBoxPadding)

                        // Returns the landmark coordinates in the input image scale.
                        return landmarkProjectionCalculator(scaleLandmark, normalizedHandRect)
                    }
                }
            }
        return emptyList()
    }

    /**
     * Decode landmark tensors into a list of landmark.
     */
    private fun tensorsToLandmarksCalculator(landmarkTensor: TensorBuffer): MutableList<HandLandmark> {
        val numValue = landmarkTensor.shape[1]
        val numDimension = numValue / NUM_LANDMARK
        val outputLandmark = mutableListOf<HandLandmark>()
        for (landmarkTypeIndex in 0 until NUM_LANDMARK) {
            val offset = landmarkTypeIndex * numDimension
            val setX = landmarkTensor.floatArray[offset]
            var setY = 0f
            var setZ = 0f
            if (numDimension > 1) {
                setY = landmarkTensor.floatArray[offset + 1]
            }
            if (numDimension > 2) {
                setZ = landmarkTensor.floatArray[offset + 2]
            }
            outputLandmark.add(HandLandmark(landmarkTypeIndex, setX, setY, setZ))
        }
        val normalizedLandmark = mutableListOf<HandLandmark>()
        outputLandmark.forEach {
            normalizedLandmark.add(
                HandLandmark(
                    it.type,
                    it.x / INPUT_WIDTH,
                    it.y / INPUT_HEIGHT,
                    it.z / INPUT_WIDTH / 0.4f // 0.4 is Z normalize
                )
            )
        }
        return normalizedLandmark
    }

    private fun convertNormalizedRectToRect(
        inputWidth: Int,
        inputHeight: Int,
        normalizedRect: Rect
    ): Rect {
        return Rect(
            normalizedRect.centerX * inputWidth,
            normalizedRect.centerY * inputHeight,
            normalizedRect.width * inputWidth,
            normalizedRect.height * inputHeight,
            normalizedRect.rotation
        )
    }

    /**
     * Convert landmarks location on the letterbox image to the corresponding locations
     * on the original image before transformation
     */
    private fun landmarkLetterBoxRemoval(
        landmarks: List<HandLandmark>,
        letterBoxPadding: FloatArray
    ): MutableList<HandLandmark> {
        val left = letterBoxPadding[0]
        val top = letterBoxPadding[1]
        val leftAndRight = letterBoxPadding[0] + letterBoxPadding[2]
        val topAndBottom = letterBoxPadding[1] + letterBoxPadding[3]
        val newNormalizeLandmark = mutableListOf<HandLandmark>()
        landmarks.forEach {
            val newX = (it.x - left) / (1f - leftAndRight)
            val newY = (it.y - top) / (1f - topAndBottom)
            val newZ = it.z / (1f - leftAndRight)
            newNormalizeLandmark.add(HandLandmark(it.type, newX, newY, newZ))
        }
        return newNormalizeLandmark
    }

    /**
     * Projects normalized landmarks in a rectangle to its original coordinates.
     * The rectangle must also be in normalized coordinates.
     */
    private fun landmarkProjectionCalculator(
        scaleLandmark: List<HandLandmark>,
        normalizedRect: Rect
    ): List<HandLandmark> {
        val originLandmark = mutableListOf<HandLandmark>()
        scaleLandmark.forEach {
            val x = it.x - 0.5f
            val y = it.y - 0.5f
            val angel = normalizedRect.rotation

            var newX = cos(angel) * x - sin(angel) * y
            var newY = sin(angel) * x + cos(angel) * y

            newX = newX * normalizedRect.width + normalizedRect.centerX
            newY = newY * normalizedRect.height + normalizedRect.centerY
            val newZ = it.z * normalizedRect.width
            originLandmark.add(HandLandmark(it.type, newX, newY, newZ))
        }
        return originLandmark
    }
}
