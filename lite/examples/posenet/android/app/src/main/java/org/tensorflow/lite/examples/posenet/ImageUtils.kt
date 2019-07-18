package org.tensorflow.lite.examples.posenet

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

import android.graphics.Bitmap
import android.os.Environment
import android.util.Log
import java.io.File
import java.io.FileOutputStream

/** Utility class for manipulating images.  */
object ImageUtils {
  // This value is 2 ^ 18 - 1, and is used to hold the RGB values together before their ranges
  // are normalized to eight bits.
  internal val kMaxChannelValue = 262143

  /** Directory where the bitmaps are saved for analysis.  */
  fun rootDirectory(): String {
    val root =
      Environment.getExternalStorageDirectory().absolutePath + File.separator +
        "tensorflow"
    return root
  }

  /**
   * Saves a Bitmap object to disk for analysis.
   *
   * @param bitmap The bitmap to save.
   * @param filename The location to save the bitmap to.
   */
  @JvmOverloads
  fun saveBitmap(bitmap: Bitmap, filename: String = "preview.png") {
    val root = rootDirectory()
    val myDir = File(root)
    if (!myDir.exists() or !myDir.isDirectory()) {
      if (!myDir.mkdirs()) {
        Log.e("Local storage", "Failed to create directory for the app in root")
      }
    }

    val file = File(myDir, filename)
    if (file.exists()) {
      file.delete()
    }
    val out = FileOutputStream(file)
    try {
      bitmap.compress(Bitmap.CompressFormat.PNG, 99, out)
      out.flush()
    } catch (e: Exception) {
      Log.e("Compressing output", e.toString())
    } finally {
      out.close()
    }
  }

  /** Helper function to convert y,u,v integer values to RGB format */
  private fun convertYUVToRGB(y: Int, u: Int, v: Int): Int {
    // Adjust and check YUV values
    var y_new = if (y - 16 < 0) 0 else y - 16
    var u_new = u - 128
    var v_new = v - 128
    val expand_y = 1192 * y_new
    var r = expand_y + 1634 * v_new
    var g = expand_y - 833 * v_new - 400 * u_new
    var b = expand_y + 2066 * u_new

    // Clipping RGB values to be inside boundaries [ 0 , kMaxChannelValue ]
    val check_boundaries = { x: Int ->
      if (x > kMaxChannelValue) kMaxChannelValue else if (x < 0) 0
      else x
    }
    r = check_boundaries(r)
    g = check_boundaries(g)
    b = check_boundaries(b)
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
    var output_index = 0
    for (j in 0 until height) {
      val position_Y = yRowStride * j
      val position_UV = uvRowStride * (j shr 1)

      for (i in 0 until width) {
        val uv_offset = position_UV + (i shr 1) * uvPixelStride

        // "0xff and" is used to cut off bits from following value that are higher than
        // the low 8 bits
        out[output_index] = convertYUVToRGB(
          0xff and yData[position_Y + i].toInt(), 0xff and uData[uv_offset].toInt(),
          0xff and vData[uv_offset].toInt()
        )
        output_index += 1
      }
    }
  }
}
