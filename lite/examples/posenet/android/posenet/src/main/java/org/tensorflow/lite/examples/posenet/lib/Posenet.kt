/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.tensorflow.lite.examples.posenet.lib

import android.graphics.Bitmap
import java.nio.ByteBuffer
import java.nio.ByteOrder
import org.tensorflow.lite.Interpreter

enum class BodyPart {
  NOSE,
  LEFT_EYE,
  RIGHT_EYE,
  LEFT_EAR,
  RIGHT_EAR,
  LEFT_SHOULDER,
  RIGHT_SHOULDER,
  LEFT_ELBOW,
  RIGHT_ELBOW,
  LEFT_WRIST,
  RIGHT_WRIST,
  LEFT_HIP,
  RIGHT_HIP,
  LEFT_KNEE,
  RIGHT_KNEE,
  LEFT_ANKLE,
  RIGHT_ANKLE
}

class Position {
  var x: Int = 0
  var y: Int = 0
}

class KeyPoint {
  var bodyPart: BodyPart = BodyPart.NOSE
  var position: Position = Position()
  var score: Float = 0.0f
}

class Person {
  var keyPoints: List<KeyPoint> = listOf<KeyPoint>()
  var score: Float = 0.0f
}

class Posenet {

  /**
   * Scale the image to a byteBuffer of [-1,1] values.
   */
  private fun initInputArray(bitmap: Bitmap): ByteBuffer {
    val bytesPerChannel = 4
    val inputChannels = 3
    val batchSize = 1
    val inputBuffer = ByteBuffer.allocateDirect(
      batchSize * bytesPerChannel * bitmap.height * bitmap.width * inputChannels
    )
    inputBuffer.order(ByteOrder.nativeOrder())
    inputBuffer.rewind()

    val mean = 128.0f
    val std = 128.0f
    for (row in 0 until bitmap.height) {
      for (col in 0 until bitmap.width) {
        var pixelValue = bitmap.getPixel(col, row)
        inputBuffer.putFloat(((pixelValue shr 16 and 0xFF) - mean) / std)
        inputBuffer.putFloat(((pixelValue shr 8 and 0xFF) - mean) / std)
        inputBuffer.putFloat(((pixelValue and 0xFF) - mean) / std)
      }
    }
    return inputBuffer
  }

  /**
   * Initializes an outputMap of 1 * x * y * z FloatArrays for the model processing to populate.
   */
  private fun initOutputMap(interpreter: Interpreter): HashMap<Int, Any> {
    var outputMap = HashMap<Int, Any>()
    // 1 * 45 * 33 * 24 contains raw part heatmaps of keypoints
    val rawPartHeatmapsShape = interpreter.getOutputTensor(0).shape()
    outputMap[0] = Array(rawPartHeatmapsShape[0]) {
      Array(rawPartHeatmapsShape[1]) {
        Array(rawPartHeatmapsShape[2]) { FloatArray(rawPartHeatmapsShape[3]) }
      }
    }

    // 1 * 45 * 33 * 17 contains heatmaps of keypoints
    val heatmapsShape = interpreter.getOutputTensor(1).shape()
    outputMap[1] = Array(heatmapsShape[0]) {
      Array(heatmapsShape[1]) { Array(heatmapsShape[2]) { FloatArray(heatmapsShape[3]) } }
    }

    // 1 * 45 * 33 * 34 contains point offsets of keypoints
    val offsetsShape = interpreter.getOutputTensor(2).shape()
    outputMap[2] = Array(heatmapsShape[0]) {
      Array(offsetsShape[1]) { Array(offsetsShape[2]) { FloatArray(offsetsShape[3]) } }
    }

    // 1 * 45 * 33 * 1 contains raw segments of keypoints
    val rawSegmentsShape = interpreter.getOutputTensor(3).shape()
    outputMap[3] = Array(rawSegmentsShape[0]) {
      Array(rawSegmentsShape[1]) { Array(rawSegmentsShape[2]) { FloatArray(rawSegmentsShape[3]) } }
    }

    return outputMap
  }

  /**
   * Estimates the pose for a single person.
   * args:
   *      interpreter: tensorflow interpreter for the Posenet model
   *      bitmap: image bitmap of frame that should be processed
   * returns:
   *      person: a person object containing data about keypoint locations and confidence scores
   */
  fun estimateSinglePose(interpreter: Interpreter, bitmap: Bitmap): Person {
    val inputArray = arrayOf(initInputArray(bitmap))
    val outputMap = initOutputMap(interpreter)
    interpreter.runForMultipleInputsOutputs(inputArray, outputMap)

    val outputRawHeatmaps = outputMap.get(1) as Array<Array<Array<FloatArray>>>
    val outputRawOffsets = outputMap.get(2) as Array<Array<Array<FloatArray>>>

    val height = outputRawHeatmaps[0].size
    val width = outputRawHeatmaps[0][0].size
    val numKeypoints = outputRawHeatmaps[0][0][0].size

    // Finds the (row, col) locations of where the keypoints are most likely to be.
    var keypointPositions = Array(numKeypoints) { Pair(0, 0) }
    for (keypoint in 0 until numKeypoints) {
      var maxVal = outputRawHeatmaps[0][0][0][keypoint]
      var maxRow = 0
      var maxCol = 0
      for (row in 0 until height) {
        for (col in 0 until width) {
          if (outputRawHeatmaps[0][row][col][keypoint] > maxVal) {
            maxVal = outputRawHeatmaps[0][row][col][keypoint]
            maxRow = row
            maxCol = col
          }
        }
      }
      keypointPositions[keypoint] = Pair(maxRow, maxCol)
    }

    // Calculating the x and y coordinates of the keypoints with offset adjustment.
    var xCoords = IntArray(numKeypoints)
    var yCoords = IntArray(numKeypoints)
    keypointPositions.forEachIndexed { idx, position ->
      yCoords[idx] = (
        position.first / (height - 1).toFloat() * bitmap.height +
          outputRawOffsets[0][keypointPositions[idx].first][keypointPositions[idx].second][idx]
        ).toInt()
      xCoords[idx] = (
        position.second / (width - 1).toFloat() * bitmap.width +
          outputRawOffsets[0][keypointPositions[idx].first]
          [keypointPositions[idx].second][idx + numKeypoints]
        ).toInt()
    }

    var person = Person()
    var keypointList = Array(numKeypoints) { KeyPoint() }
    enumValues<BodyPart>().forEachIndexed { idx, it ->
      keypointList[idx].bodyPart = it
      keypointList[idx].position.x = xCoords[idx]
      keypointList[idx].position.y = yCoords[idx]
    }

    // TODO(eileenmao):ignore the unknown keypoint and get confidence scores
    person.keyPoints = keypointList.toList()

    return person
  }
}
