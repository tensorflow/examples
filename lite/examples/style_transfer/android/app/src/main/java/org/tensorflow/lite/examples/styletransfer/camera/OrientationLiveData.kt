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

package org.tensorflow.lite.examples.styletransfer.camera

import android.content.Context
import android.view.OrientationEventListener
import android.view.Surface
import androidx.lifecycle.LiveData

/**
 * Calculates closest 90-degree orientation to compensate for the device
 * rotation relative to sensor orientation, i.e., allows user to see camera
 * frames with the expected orientation.
 */
class OrientationLiveData(
  context: Context
) : LiveData<Int>() {

  private val listener = object : OrientationEventListener(context.applicationContext) {
    override fun onOrientationChanged(orientation: Int) {
      val rotation = when {
        orientation <= 45 -> Surface.ROTATION_0
        orientation <= 135 -> Surface.ROTATION_90
        orientation <= 225 -> Surface.ROTATION_180
        orientation <= 315 -> Surface.ROTATION_270
        else -> Surface.ROTATION_0
      }
       postValue(rotation)
    }
  }

  override fun onActive() {
    super.onActive()
    listener.enable()
  }

  override fun onInactive() {
    super.onInactive()
    listener.disable()
  }
}
