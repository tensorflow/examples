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

package org.tensorflow.lite.examples.soundclassifier.compose.ui

import android.annotation.SuppressLint
import android.app.Application
import android.media.AudioRecord
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import androidx.core.os.HandlerCompat
import androidx.lifecycle.AndroidViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import org.tensorflow.lite.support.label.Category
import org.tensorflow.lite.task.audio.classifier.AudioClassifier

const val DefaultClassificationInterval = 500L

@SuppressLint("StaticFieldLeak")
class SoundClassifierViewModel(application: Application) : AndroidViewModel(application) {
  // Changing this value triggers turning classification on/off
  private val _classifierEnabled = MutableStateFlow(true)
  val classifierEnabled = _classifierEnabled.asStateFlow()

  // How often should classification run in milliseconds
  private val _classificationInterval = MutableStateFlow(DefaultClassificationInterval)
  val classificationInterval = _classificationInterval.asStateFlow()

  // As a result of sound classification, this value emits map of probabilities
  private val _probabilities = MutableStateFlow<List<Category>>(emptyList())
  val probabilities = _probabilities.asStateFlow()

  private var handler: Handler // background thread handler to run classification
  private var audioClassifier: AudioClassifier? = null
  private var audioRecord: AudioRecord? = null

  init {
    // Create a handler to run classification in a background thread
    val handlerThread = HandlerThread("backgroundThread")
    handlerThread.start()
    handler = HandlerCompat.createAsync(handlerThread.looper)
  }

  fun setClassifierEnabled(value: Boolean) {
    _classifierEnabled.value = value
  }

  fun setClassificationInterval(value: Long) {
    _classificationInterval.value = value
  }

  fun setProbabilities(value: List<Category>) {
    _probabilities.value = value
  }

  fun startAudioClassification() {
    // If the audio classifier is initialized and running, do nothing.
    if (audioClassifier != null) {
      setClassifierEnabled(true)
      return
    }

    // Initialize the audio classifier
    val classifier = AudioClassifier.createFromFile(getApplication(), MODEL_FILE)
    val audioTensor = classifier.createInputTensorAudio()

    // Initialize the audio recorder
    val record = classifier.createAudioRecord()
    record.startRecording()

    // Define the classification runnable
    val run = object : Runnable {
      override fun run() {
        val startTime = System.currentTimeMillis()

        // Load the latest audio sample
        audioTensor.load(record)
        val output = classifier.classify(audioTensor)

        // Filter out results above a certain threshold, and sort them descendingly
        val filteredModelOutput = output[0].categories.filter {
          it.score > MINIMUM_DISPLAY_THRESHOLD
        }.sortedBy {
          -it.score
        }
        val finishTime = System.currentTimeMillis()
        Log.d(LOG_TAG, "Latency = ${finishTime - startTime} ms")

        setProbabilities(filteredModelOutput)

        // Rerun the classification after a certain interval
        handler.postDelayed(this, classificationInterval.value)
      }
    }

    // Start the classification process
    handler.post(run)

    // Save the instances we just created for use later
    audioClassifier = classifier
    audioRecord = record
  }

  fun stopAudioClassification() {
    handler.removeCallbacksAndMessages(null)
    audioRecord?.stop()
    audioRecord = null
    audioClassifier = null
  }

  companion object {
    private const val LOG_TAG = "AudioDemo"
    private const val MODEL_FILE = "yamnet.tflite"
    private const val MINIMUM_DISPLAY_THRESHOLD: Float = 0.3f
  }
}
