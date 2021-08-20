package org.tensorflow.lite.examples.soundclassifier.compose

import android.Manifest
import android.content.pm.PackageManager
import android.media.AudioRecord
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.annotation.RequiresApi
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.core.content.ContextCompat
import androidx.core.os.HandlerCompat
import kotlinx.coroutines.flow.MutableStateFlow
import org.tensorflow.lite.examples.soundclassifier.compose.ui.SoundClassifierScene
import org.tensorflow.lite.support.label.Category
import org.tensorflow.lite.task.audio.classifier.AudioClassifier

class MainActivity : ComponentActivity() {
  private val classifierEnabled: Boolean
    get() = (audioClassifier != null)

  private var audioClassifier: AudioClassifier? = null
  private var audioRecord: AudioRecord? = null

  // TODO move this to ViewModel
  private var classificationInterval = 500L // how often should classification run in milli-secs
  private lateinit var handler: Handler // background thread handler to run classification

  /** As a result of sound classification, this value emits map of probabilities */
  private val probabilities = MutableStateFlow<List<Category>>(emptyList())

  private val requestAudioPermissionLauncher = registerForActivityResult(
    ActivityResultContracts.RequestPermission()
  ) { isGranted ->
    if (isGranted) {
      Log.i(LOG_TAG, "Audio permission granted")
      startAudioClassification()
    } else {
      Log.w(LOG_TAG, "Audio permission not granted!")
    }
  }

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)

    setContent {
      var overlapFactor by rememberSaveable { mutableStateOf(0.8f) }

      SoundClassifierScene(
        probabilities = probabilities,
        classifierEnabled = classifierEnabled,
        interval = classificationInterval,
        onClassifierToggle = { enabled: Boolean ->
          if (enabled) startAudioClassification() else stopAudioClassification()
          keepScreenOn(enabled)
        },
        onIntervalChanged = { value ->
          classificationInterval = value
        }
      )
    }

    keepScreenOn(classifierEnabled)

    // Create a handler to run classification in a background thread
    val handlerThread = HandlerThread("backgroundThread")
    handlerThread.start()
    handler = HandlerCompat.createAsync(handlerThread.looper)

    // Request microphone permission and start running classification
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
      requestMicrophonePermission()
    } else {
      startAudioClassification()
    }
  }

  private fun startAudioClassification() {
    // If the audio classifier is initialized and running, do nothing.
    if (audioClassifier != null) return

    // Initialize the audio classifier
    val classifier = AudioClassifier.createFromFile(this, MODEL_FILE)
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

        probabilities.value = filteredModelOutput

        // Rerun the classification after a certain interval
        handler.postDelayed(this, classificationInterval)
      }
    }

    // Start the classification process
    handler.post(run)

    // Save the instances we just created for use later
    audioClassifier = classifier
    audioRecord = record
  }

  private fun stopAudioClassification() {
    handler.removeCallbacksAndMessages(null)
    audioRecord?.stop()
    audioRecord = null
    audioClassifier = null
  }

  override fun onTopResumedActivityChanged(isTopResumedActivity: Boolean) {
    // Handles "top" resumed event on multi-window environment
    if (isTopResumedActivity) {
      requestMicrophonePermission()
    } else {
      stopAudioClassification()
    }
  }

  @RequiresApi(Build.VERSION_CODES.M)
  private fun requestMicrophonePermission() {
    if (ContextCompat.checkSelfPermission(
        this,
        Manifest.permission.RECORD_AUDIO
      ) == PackageManager.PERMISSION_GRANTED
    ) {
      startAudioClassification()
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
    private const val MODEL_FILE = "yamnet.tflite"
    private const val MINIMUM_DISPLAY_THRESHOLD: Float = 0.3f
  }
}
