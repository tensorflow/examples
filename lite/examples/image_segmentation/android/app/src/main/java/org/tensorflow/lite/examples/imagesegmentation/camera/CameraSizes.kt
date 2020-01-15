/*
 * Copyright 2019 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.imagesegmentation.camera

import android.graphics.Point
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.params.StreamConfigurationMap
import android.util.Size
import android.view.Display
import kotlin.math.max
import kotlin.math.min

/** Helper class used to pre-compute shortest and longest sides of a [Size] */
class SmartSize(width: Int, height: Int) {
  var size = Size(width, height)
  var long = max(size.width, size.height)
  var short = min(size.width, size.height)
  override fun toString() = "SmartSize(${long}x$short)"
}

/** Standard High Definition size for pictures and video */
val SIZE_1080P: SmartSize = SmartSize(1920, 1080)

/** Returns a [SmartSize] object for the given [Display] */
fun getDisplaySmartSize(display: Display): SmartSize {
  val outPoint = Point()
  display.getRealSize(outPoint)
  return SmartSize(outPoint.x, outPoint.y)
}

// verify that the given width and height are on the expected aspect ratio
fun verifyAspectRatio(width: Int, height: Int, aspectRatio: Size): Boolean {
  return (width * aspectRatio.height) == (height * aspectRatio.width)
}

/**
 * Returns the largest available PREVIEW size. For more information, see:
 * https://d.android.com/reference/android/hardware/camera2/CameraDevice
 */
fun <T> getPreviewOutputSize(
  display: Display,
  characteristics: CameraCharacteristics,
  targetClass: Class<T>,
  aspectRatio: Size,
  format: Int? = null
): Size {
  // Find which is smaller: screen or 1080p
  val screenSize = getDisplaySmartSize(display)
  val hdScreen = screenSize.long >= SIZE_1080P.long || screenSize.short >= SIZE_1080P.short
  val maxSize = if (hdScreen) SIZE_1080P else screenSize

  // If image format is provided, use it to determine supported sizes; else use target class
  val config = characteristics.get(
    CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP
  )!!
  if (format == null) {
    assert(StreamConfigurationMap.isOutputSupportedFor(targetClass))
  } else {
    assert(config.isOutputSupportedFor(format))
  }
  val allSizes = if (format == null) {
    config.getOutputSizes(targetClass)
  } else {
    config.getOutputSizes(format)
  }

  // Get available sizes and sort them by area from largest to smallest
  val validSizes = allSizes
    .sortedWith(compareBy { it.height * it.width })
    .filter { verifyAspectRatio(it.width, it.height, aspectRatio) }
    .map { SmartSize(it.width, it.height) }.reversed()

  // Then, get the largest output size that is smaller or equal than our max size
  return validSizes.first { it.long <= maxSize.long && it.short <= maxSize.short }.size
}
