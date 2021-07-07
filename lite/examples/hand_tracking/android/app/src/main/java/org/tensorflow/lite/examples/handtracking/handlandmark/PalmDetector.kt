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
import android.graphics.RectF
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.examples.handtracking.handlandmark.data.*
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.lang.RuntimeException
import java.nio.Buffer
import kotlin.math.*


class PalmDetector(private val interpreter: Interpreter) {
    companion object {
        private const val MODEL_PATH = "palm_detection.tflite"
        private val strides = intArrayOf(8, 16, 16, 16)
        private val rawBoxShape = intArrayOf(1, 896, 18)
        private val rawScoreShape = intArrayOf(1, 896, 1)
        private const val IMAGE_MEAN = 127.5f
        private const val IMAGE_STD = 127.5f
        private const val INPUT_WIDTH = 128
        private const val INPUT_HEIGHT = 128
        private const val NUM_BOX = 896
        private const val NUM_CLASSES = 1
        private const val NUM_LAYER = 4
        private const val ANCHOR_OFFSET_X = 0.5f
        private const val ANCHOR_OFFSET_Y = 0.5f
        private const val ANCHOR_MIN_SCALE = 0.1484375f
        private const val ANCHOR_MAX_SCALE = 0.75f
        private const val NUM_COORDS = 18
        private const val KEYPOINT_COORD_OFFSET = 4
        private const val NUM_VALUE_PER_KEYPOINT = 2
        private const val NUM_KEYPOINT = 7
        private const val BOX_COORD_OFFSET = 0
        private const val MIN_SCORE_THRESH = 0.5f
        private const val MIN_SUPPRESSION_THRESHOLD = 0.3f
        private const val SCORE_CLIPPING_THRESH = 100f

        fun create(context: Context): PalmDetector {
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
            return PalmDetector(interpreter)
        }
    }

    // Detect palm and return the detection results
    fun process(inputImage: Bitmap): Detection? {
        val letterBoxPadding =
            DetectionHelper.getLetterBoxPadding(
                inputImage.width,
                inputImage.height,
                INPUT_WIDTH,
                INPUT_HEIGHT
            )
        // resize and normalize input image
        val imageProcessor = ImageProcessor.Builder()
            .add(
                ResizeOp(
                    INPUT_WIDTH,
                    INPUT_HEIGHT,
                    ResizeOp.ResizeMethod.BILINEAR
                )
            )
            .add(
                NormalizeOp(
                    IMAGE_MEAN,
                    IMAGE_STD
                )
            )
            .build()
        val inputTensorImage =
            imageProcessor.process(TensorImage(DataType.FLOAT32).apply { load(inputImage) })
        val rawBoxTensor =
            TensorBuffer.createFixedSize(rawBoxShape, DataType.FLOAT32)
        val rawScoreTensor =
            TensorBuffer.createFixedSize(rawScoreShape, DataType.FLOAT32)
        val outputs: Map<Int, Buffer> =
            mapOf(0 to rawBoxTensor.buffer.rewind(), 1 to rawScoreTensor.buffer.rewind())
        interpreter.runForMultipleInputsOutputs(
            arrayOf(inputTensorImage.tensorBuffer.buffer),
            outputs
        )
        val anchor = createAnchor()
        val boxes = decodeBoxes(rawBoxTensor.floatArray, anchor)
        val unfilteredDetections = tensorToDetection(boxes, rawScoreTensor.floatArray)
        val filteredDetections = weightNonMaxSuppression(unfilteredDetections)
        val originalDetection = detectionLetterboxRemove(filteredDetections, letterBoxPadding)
        if (originalDetection.size >= 1) {
            // return only one detection object
            return originalDetection[0]
        }
        // return null if no palm detected.
        return null
    }

    // Releases resources if no longer used.
    fun close() {
        interpreter.close()
    }

    /**
     * Generate a list of SSD anchor
     */
    private fun createAnchor(): List<Anchor> {
        val aspectRatioInit = listOf(1f)
        val anchors = mutableListOf<Anchor>()
        var layerId = 0
        while (layerId < NUM_LAYER) {
            val anchorHeight = mutableListOf<Float>()
            val aspectRatios = mutableListOf<Float>()
            val scales = mutableListOf<Float>()
            var lastSameStrideLayer = layerId
            while (lastSameStrideLayer < strides.size && strides[lastSameStrideLayer] == strides[layerId]) {
                val scale = calculatorScale(lastSameStrideLayer, strides.size)
                for (element in aspectRatioInit) {
                    aspectRatios.add(element)
                    scales.add(scale)
                }
                val scaleNext =
                    if (lastSameStrideLayer == strides.size - 1) 1f else calculatorScale(
                        lastSameStrideLayer + 1,
                        strides.size
                    )
                scales.add(sqrt(scale * scaleNext))
                aspectRatios.add(1f)
                lastSameStrideLayer++
            }
            for (i in 0 until aspectRatios.size) {
                val ratioSqrt = sqrt(aspectRatios[i])
                anchorHeight.add(scales[i] / ratioSqrt)
            }
            val stride = strides[layerId]
            val featureMapHeight: Int = ceil(1f * INPUT_HEIGHT / stride).toInt()
            val featureMapWidth: Int = ceil(1f * INPUT_WIDTH / stride).toInt()
            for (y in 0 until featureMapHeight) {
                for (x in 0 until featureMapWidth) {
                    for (anchorId in 0 until anchorHeight.size) {
                        val centerX = (x + ANCHOR_OFFSET_X) * 1f / featureMapWidth
                        val centerY = (y + ANCHOR_OFFSET_Y) * 1f / featureMapHeight
                        val anchor = Anchor(
                            xCenter = centerX,
                            yCenter = centerY,
                            h = 1f,
                            w = 1f
                        )
                        anchors.add(anchor)
                    }
                }
            }
            layerId = lastSameStrideLayer
        }
        return anchors
    }

    private fun calculatorScale(
        strideIndex: Int,
        numStride: Int
    ): Float {
        return if (numStride == 1) {
            (ANCHOR_MIN_SCALE + ANCHOR_MAX_SCALE) * 0.5f
        } else {
            ANCHOR_MIN_SCALE + (ANCHOR_MAX_SCALE - ANCHOR_MIN_SCALE) * 1f * strideIndex / (numStride - 1f)
        }
    }

    /**
     * Convert tensor raw boxes to detection boxes
     */
    private fun decodeBoxes(rawBoxes: FloatArray, anchor: List<Anchor>): FloatArray {
        val boxes = FloatArray(NUM_BOX * NUM_COORDS)
        for (i in 0 until NUM_BOX) {
            val boxOffset = i * NUM_COORDS + BOX_COORD_OFFSET
            var xCenter = rawBoxes[boxOffset]
            var yCenter = rawBoxes[boxOffset + 1]
            var w = rawBoxes[boxOffset + 2]
            var h = rawBoxes[boxOffset + 3]
            xCenter = xCenter / INPUT_WIDTH * anchor[i].w + anchor[i].xCenter
            yCenter = yCenter / INPUT_HEIGHT * anchor[i].h + anchor[i].yCenter
            h = h / INPUT_HEIGHT * anchor[i].h
            w = w / INPUT_WIDTH * anchor[i].w
            val yMin = yCenter - h / 2f
            val xMin = xCenter - w / 2f
            val yMax = yCenter + h / 2f
            val xMax = xCenter + w / 2f
            boxes[i * NUM_COORDS + 0] = yMin
            boxes[i * NUM_COORDS + 1] = xMin
            boxes[i * NUM_COORDS + 2] = yMax
            boxes[i * NUM_COORDS + 3] = xMax
            for (k in 0 until NUM_KEYPOINT) {
                val offset = i * NUM_COORDS + KEYPOINT_COORD_OFFSET + k * NUM_VALUE_PER_KEYPOINT
                val keyPointX = rawBoxes[offset]
                val keyPointY = rawBoxes[offset + 1]
                boxes[offset] = keyPointX / INPUT_WIDTH * anchor[i].w + anchor[i].xCenter
                boxes[offset + 1] = keyPointY / INPUT_HEIGHT * anchor[i].h + anchor[i].yCenter
            }
        }
        return boxes
    }

    /**
     * Convert the detection tensors to list detection object
     */
    private fun tensorToDetection(
        boxes: FloatArray,
        rawScores: FloatArray
    ): List<Detection> {
        val detectionScores = FloatArray(NUM_BOX)
        val detectionClasses = IntArray(NUM_BOX)
        for (i in 0 until NUM_BOX) {
            var classId = -1
            var maxScore = -Float.MAX_VALUE
            for (scoreIdx in 0 until NUM_CLASSES) {
                var score = rawScores[i * NUM_CLASSES + scoreIdx]
                score = if (score < -SCORE_CLIPPING_THRESH) -SCORE_CLIPPING_THRESH else score
                score = if (score > SCORE_CLIPPING_THRESH) SCORE_CLIPPING_THRESH else score
                score = 1f / (1f + exp(-score))
                if (maxScore < score) {
                    maxScore = score
                    classId = scoreIdx
                }
            }
            detectionScores[i] = maxScore
            detectionClasses[i] = classId
        }
        return convertToDetections(boxes, detectionScores, detectionClasses)
    }

    private fun convertToDetections(
        boxes: FloatArray,
        detectionScores: FloatArray,
        detectionClasses: IntArray
    ): List<Detection> {
        val detections = mutableListOf<Detection>()
        for (i in 0 until NUM_BOX) {
            if (detectionScores[i] < MIN_SCORE_THRESH) continue
            val boxOffset = i * NUM_COORDS
            val detection = convertToDetection(
                boxes[boxOffset + 0],
                boxes[boxOffset + 1],
                boxes[boxOffset + 2],
                boxes[boxOffset + 3],
                detectionScores[i],
                detectionClasses[i]
            )
            val bbox = detection.locationData.relativeBoundingBox
            if (bbox.width < 0 || bbox.height < 0) continue
            // Add key points
            for (kp_id in 0 until (NUM_KEYPOINT * NUM_VALUE_PER_KEYPOINT) step NUM_VALUE_PER_KEYPOINT) {
                val keyPoint = RelativeKeyPoints()
                val keyPointIndex = boxOffset + KEYPOINT_COORD_OFFSET + kp_id
                keyPoint.x = boxes[keyPointIndex + 0]
                keyPoint.y = boxes[keyPointIndex + 1]
                detection.locationData.relativeKeyPoints.add(keyPoint)
            }
            detections.add(detection)
        }
        return detections
    }

    /**
     * Convert data into detection object.
     */
    private fun convertToDetection(
        boxYMin: Float,
        boxXMin: Float,
        boxYMax: Float,
        boxXMax: Float,
        score: Float,
        classId: Int
    ): Detection = Detection(
        score = score,
        labelId = classId,
        locationData = LocationData(
            relativeBoundingBox = RelativeBoundingBox(
                boxXMin,
                boxYMin,
                boxXMax - boxXMin,
                boxYMax - boxYMin
            )
        )
    )

    /**
     * Performs non-max suppression to remove excessive detections
     */
    private fun weightNonMaxSuppression(
        detections: List<Detection>
    ): MutableList<Detection> {
        val outputDetections = mutableListOf<Detection>()
        var remainedIndexedScores = mutableListOf<Pair<Int, Float>>()
        detections.forEachIndexed { index, detection ->
            remainedIndexedScores.add(Pair(index, detection.score))
        }
        remainedIndexedScores.sortBy { it.second }
        val remained = mutableListOf<Pair<Int, Float>>()
        val candidates = mutableListOf<Pair<Int, Float>>()
        while (remainedIndexedScores.isNotEmpty()) {
            val originalIndexedScoresSize = remainedIndexedScores.size
            val detection = detections[remainedIndexedScores[0].first]
            remained.clear()
            candidates.clear()
            val location = detection.locationData
            remainedIndexedScores.forEach { indexScore ->
                val restLocation = detections[indexScore.first].locationData
                val similarity = overlapSimilarity(
                    restLocation.getRelativeBoundingBox(),
                    location.getRelativeBoundingBox()
                )
                if (similarity > MIN_SUPPRESSION_THRESHOLD) {
                    candidates.add(indexScore)
                } else {
                    remained.add(indexScore)
                }
            }

            if (candidates.isNotEmpty()) {
                val numKeyPoints = detection.locationData.relativeKeyPoints.size
                val keyPoints = MutableList(numKeyPoints * 2) { 0f }
                var wXMin = 0f
                var wYMin = 0f
                var wXMax = 0f
                var wYMax = 0f
                var totalScore = 0f
                candidates.forEach { candidate ->
                    totalScore += candidate.second
                    val locationData = detections[candidate.first].locationData
                    locationData.relativeBoundingBox.let { bbox ->
                        wXMin += bbox.xMin * candidate.second
                        wYMin += bbox.yMin * candidate.second
                        wXMax += (bbox.xMin + bbox.width) * candidate.second
                        wYMax += (bbox.yMin + bbox.height) * candidate.second
                    }
                    for (i in 0 until numKeyPoints) {
                        keyPoints[i * 2] += locationData.relativeKeyPoints[i].x * candidate.second
                        keyPoints[i * 2 + 1] += locationData.relativeKeyPoints[i].y * candidate.second
                    }
                }
                val weightLocation = detection.locationData.relativeBoundingBox
                weightLocation.run {
                    xMin = wXMin / totalScore
                    yMin = wYMin / totalScore
                    width = (wXMax / totalScore) - weightLocation.xMin
                    height = (wYMax / totalScore) - weightLocation.yMin
                }
                for (i in 0 until numKeyPoints) {
                    val keyPoint = detection.locationData.relativeKeyPoints[i]
                    keyPoint.x = keyPoints[i * 2] / totalScore
                    keyPoint.y = keyPoints[i * 2 + 1] / totalScore
                }
            }
            outputDetections.add(detection)
            if (originalIndexedScoresSize == remained.size) {
                break
            } else {
                remainedIndexedScores = remained
            }
        }
        return outputDetections
    }

    private fun overlapSimilarity(rect1: RectF, rect2: RectF): Float {
        if (!rect1.intersect(rect2)) return 0f
        val xOverlap = max(0f, min(rect1.right, rect2.right) - max(rect1.left, rect2.left))
        val yOverlap = max(0f, min(rect1.bottom, rect2.bottom) - max(rect1.top, rect2.top))
        val intersectionArea = xOverlap * yOverlap
        val normalization =
            (rect1.width() * rect1.height()) + (rect2.width() * rect2.height()) - intersectionArea
        return if (normalization > 0f) intersectionArea / normalization else 0f
    }

    /**
     * Converts detection locations on the letterbox image to the corresponding locations
     * on the original image before transformation
     */
    private fun detectionLetterboxRemove(
        detections: MutableList<Detection>,
        letterBoxPadding: FloatArray
    ): MutableList<Detection> {
        val left = letterBoxPadding[0]
        val top = letterBoxPadding[1]
        val leftAndRight = letterBoxPadding[0] + letterBoxPadding[2]
        val topAndBottom = letterBoxPadding[1] + letterBoxPadding[3]
        detections.forEach { detection ->
            detection.locationData.relativeBoundingBox.let {
                it.xMin = (it.xMin - left) / (1f - leftAndRight)
                it.yMin = (it.yMin - top) / (1f - topAndBottom)

                it.width = it.width / (1f - leftAndRight)
                it.height = it.height / (1f - topAndBottom)
            }
            detection.locationData.relativeKeyPoints.forEach {
                val newX = (it.x - left) / (1f - leftAndRight)
                val newY = (it.y - top) / (1f - topAndBottom)
                it.x = newX
                it.y = newY
            }
        }
        return detections
    }
}
