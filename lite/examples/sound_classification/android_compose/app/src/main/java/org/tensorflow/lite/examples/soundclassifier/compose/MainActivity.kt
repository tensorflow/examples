/*
 * Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

package org.tensorflow.lite.examples.soundclassifier.compose

import android.Manifest
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.util.Log
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.annotation.RequiresApi
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.flow.collect
import org.tensorflow.lite.examples.soundclassifier.compose.ui.SoundClassifierScreen
import org.tensorflow.lite.examples.soundclassifier.compose.ui.SoundClassifierViewModel

class MainActivity : ComponentActivity() {
  private val viewModel: SoundClassifierViewModel by viewModels()

  private val requestAudioPermissionLauncher = registerForActivityResult(
    ActivityResultContracts.RequestPermission()
  ) { isGranted ->
    if (isGranted) {
      Log.i(LOG_TAG, "Audio permission granted")
      viewModel.startAudioClassification()
    } else {
      Log.w(LOG_TAG, "Audio permission not granted!")
    }
  }

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)

    setContent {
      SoundClassifierScreen(viewModel)
    }

    lifecycleScope.launchWhenCreated {
      viewModel.classifierEnabled.collect { enabled ->
        if (enabled) {
          // Request microphone permission and start running classification
          if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            requestMicrophonePermission()
          } else {
            viewModel.startAudioClassification()
          }
        } else {
          viewModel.stopAudioClassification()
        }
        keepScreenOn(enabled)
      }
    }
  }

  override fun onTopResumedActivityChanged(isTopResumedActivity: Boolean) {
    // Handles "top" resumed event on multi-window environment
    if (isTopResumedActivity) {
      requestMicrophonePermission()
    } else {
      viewModel.stopAudioClassification()
    }
  }

  @RequiresApi(Build.VERSION_CODES.M)
  private fun requestMicrophonePermission() {
    if (ContextCompat.checkSelfPermission(
        this,
        Manifest.permission.RECORD_AUDIO
      ) == PackageManager.PERMISSION_GRANTED
    ) {
      viewModel.startAudioClassification()
    } else {
      requestAudioPermissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
    }
  }

  private fun keepScreenOn(enable: Boolean) =
    if (enable) {
      window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
    } else {
      window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
    }

  companion object {
    private const val LOG_TAG = "AudioDemo"
  }
}
