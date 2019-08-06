/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package org.tensorflow.lite.examples.posenet

/** Utility class for manipulating images.  */
object ImageUtils {
  // This value is 2 ^ 18 - 1, and is used to hold the RGB values together before their ranges
  // are normalized to eight bits.
  private const val MAX_CHANNEL_VALUE = 262143

  /** Helper function to convert y,u,v integer values to RGB format */
  private fun convertYUVToRGB(y: Int, u: Int, v: Int): Int {
    // Adjust and check YUV values
    val yNew = if (y - 16 < 0) 0 else y - 16
    val uNew = u - 128
    val vNew = v - 128
    val expandY = 1192 * yNew
    var r = expandY + 1634 * vNew
    var g = expandY - 833 * vNew - 400 * uNew
    var b = expandY + 2066 * uNew

    // Clipping RGB values to be inside boundaries [ 0 , MAX_CHANNEL_VALUE ]
    val checkBoundaries = { x: Int ->
      when {
        x > MAX_CHANNEL_VALUE -> MAX_CHANNEL_VALUE
        x < 0 -> 0
        else -> x
      }
    }
    r = checkBoundaries(r)
    g = checkBoundaries(g)
    b = checkBoundaries(b)
    return -0x1000000 or (r shl 6 and 0xff0000) or (g shr 2 and 0xff00) or (b shr 10 and 0xff)
  }

  /** Converts YUV420 format image data (ByteArray) into ARGB8888 format with IntArray as output. */
  fun convertYUV420ToARGB8888(
    yData: ByteArray,
    uData: ByteArray,
    vData: ByteArray,
    width: Int,
    height: Int,
    yRowStride: Int,
    uvRowStride: Int,
    uvPixelStride: Int,
    out: IntArray
  ) {
    var outputIndex = 0
    for (j in 0 until height) {
      val positionY = yRowStride * j
      val positionUV = uvRowStride * (j shr 1)

      for (i in 0 until width) {
        val uvOffset = positionUV + (i shr 1) * uvPixelStride

        // "0xff and" is used to cut off bits from following value that are higher than
        // the low 8 bits
        out[outputIndex] = convertYUVToRGB(
          0xff and yData[positionY + i].toInt(), 0xff and uData[uvOffset].toInt(),
          0xff and vData[uvOffset].toInt()
        )
        outputIndex += 1
      }
    }
  }
}
