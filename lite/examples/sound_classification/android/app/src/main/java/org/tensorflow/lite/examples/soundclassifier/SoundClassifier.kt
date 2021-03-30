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

package org.tensorflow.lite.examples.soundclassifier

import android.content.Context
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.SystemClock
import android.util.Log
import androidx.annotation.MainThread
import androidx.lifecycle.DefaultLifecycleObserver
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import java.io.BufferedReader
import java.io.IOException
import java.io.InputStreamReader
import java.nio.FloatBuffer
import java.util.Locale
import java.util.concurrent.TimeUnit
import kotlin.math.sin
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil

/**
 * Performs classification on sound.
 *
 * <p>The API supports models which accept sound input via {@code AudioRecord} and one classification output tensor.
 * The output of the recognition is emitted as LiveData of Map.
 *
 */
class SoundClassifier(context: Context, private val options: Options = Options()) :
  DefaultLifecycleObserver {
  class Options constructor(
    /** Path of the converted model label file, relative to the assets/ directory.  */
    val metadataPath: String = "labels.txt",
    /** Path of the converted .tflite file, relative to the assets/ directory.  */
    val modelPath: String = "sound_classifier.tflite",
    /** The required audio sample rate in Hz.  */
    val sampleRate: Int = 44_100,
    /** How many milliseconds to sleep between successive audio sample pulls.  */
    val audioPullPeriod: Long = 50L,
    /** Number of warm up runs to do after loading the TFLite model.  */
    val warmupRuns: Int = 3,
    /** Number of points in average to reduce noise. */
    val pointsInAverage: Int = 10,
    /** Overlap factor of recognition period */
    var overlapFactor: Float = 0.8f,
    /** Probability value above which a class is labeled as active (i.e., detected) the display.  */
    var probabilityThreshold: Float = 0.3f,
  )

  var isRecording: Boolean = false
    private set

  /** As the result of sound classification, this value emits map of probabilities */
  val probabilities: LiveData<Map<String, Float>>
    get() = _probabilities
  private val _probabilities = MutableLiveData<Map<String, Float>>()

  var isClosed: Boolean = true
    private set

  /**
   * LifecycleOwner instance to deal with RESUME, PAUSE and DESTROY events automatically.
   * You can also handle those events by calling `start()`, `stop()` and `close()` methods
   * manually.
   */
  var lifecycleOwner: LifecycleOwner? = null
    @MainThread
    set(value) {
      if (field === value) return
      field?.lifecycle?.removeObserver(this)
      field = value?.also {
        it.lifecycle.addObserver(this)
      }
    }

  /** Overlap factor of recognition period */
  var overlapFactor: Float
    get() = options.overlapFactor
    set(value) {
      options.overlapFactor = value.also {
        recognitionPeriod = (1000L * (1 - value)).toLong()
      }
    }

  /** Probability value above which a class is labeled as active (i.e., detected) the display.  */
  var probabilityThreshold: Float
    get() = options.probabilityThreshold
    set(value) {
      options.probabilityThreshold = value
    }

  /** Paused by user */
  var isPaused: Boolean = false
    set(value) {
      field = value
      if (value) stop() else start()
    }

  /** Names of the model's output classes.  */
  lateinit var labelList: List<String>
    private set

  /** How many milliseconds between consecutive model inference calls.  */
  private var recognitionPeriod = (1000L * (1 - overlapFactor)).toLong()

  /** The TFLite interpreter instance.  */
  private lateinit var interpreter: Interpreter

  /** Audio length (in # of PCM samples) required by the TFLite model.  */
  private var modelInputLength = 0

  /** Number of output classes of the TFLite model.  */
  private var modelNumClasses = 0

  /** Used to hold the real-time probabilities predicted by the model for the output classes.  */
  private lateinit var predictionProbs: FloatArray

  /** Latest prediction latency in milliseconds.  */
  private var latestPredictionLatencyMs = 0f

  private var recognitionThread: Thread? = null

  /** Used to record audio samples. */
  private lateinit var audioRecord: AudioRecord

  /** Buffer that holds audio PCM sample that are fed to the TFLite model for inference.  */
  private lateinit var inputBuffer: FloatBuffer

  init {
    loadLabels(context)
    setupInterpreter(context)
    warmUpModel()
  }

  override fun onResume(owner: LifecycleOwner) = start()

  override fun onPause(owner: LifecycleOwner) = stop()

  /**
   * Starts sound classification, which triggers running of
   * `recordingThread` and `recognitionThread`.
   */
  fun start() {
    if (!isPaused) {
      startAudioRecord()
    }
  }

  /**
   * Stops sound classification, which triggers interruption of
   * `recognitionThread`.
   */
  fun stop() {
    if (isClosed || !isRecording) return
    recognitionThread?.interrupt()
    audioRecord.stop()
    isRecording = false

    _probabilities.postValue(labelList.associateWith { 0f })
  }

  fun close() {
    stop()

    if (isClosed) return
    interpreter.close()

    isClosed = true
  }

  /** Retrieve labels from "labels.txt" file */
  private fun loadLabels(context: Context) {
    try {
      val reader =
        BufferedReader(InputStreamReader(context.assets.open(options.metadataPath)))
      val wordList = mutableListOf<String>()
      reader.useLines { lines ->
        lines.forEach {
          wordList.add(it.split(" ").last())
        }
      }
      labelList = wordList.map { it.toTitleCase() }
    } catch (e: IOException) {
      Log.e(TAG, "Failed to read model ${options.metadataPath}: ${e.message}")
    }
  }

  private fun setupInterpreter(context: Context) {
    interpreter = try {
      val tfliteBuffer = FileUtil.loadMappedFile(context, options.modelPath)
      Log.i(TAG, "Done creating TFLite buffer from ${options.modelPath}")
      Interpreter(tfliteBuffer, Interpreter.Options())
    } catch (e: IOException) {
      Log.e(TAG, "Failed to load TFLite model - ${e.message}")
      return
    }

    // Inspect input and output specs.
    val inputShape = interpreter.getInputTensor(0).shape()
    Log.i(TAG, "TFLite model input shape: ${inputShape.contentToString()}")
    modelInputLength = inputShape[1]

    val outputShape = interpreter.getOutputTensor(0).shape()
    Log.i(TAG, "TFLite output shape: ${outputShape.contentToString()}")
    modelNumClasses = outputShape[1]
    if (modelNumClasses != labelList.size) {
      Log.e(
        TAG,
        "Mismatch between metadata number of classes (${labelList.size})" +
          " and model output length ($modelNumClasses)"
      )
    }
    // Fill the array with NaNs initially.
    predictionProbs = FloatArray(modelNumClasses) { Float.NaN }

    inputBuffer = FloatBuffer.allocate(modelInputLength)
  }

  private fun warmUpModel() {
    generateDummyAudioInput(inputBuffer)
    for (n in 0 until options.warmupRuns) {
      val t0 = SystemClock.elapsedRealtimeNanos()

      // Create input and output buffers.
      val outputBuffer = FloatBuffer.allocate(modelNumClasses)
      inputBuffer.rewind()
      outputBuffer.rewind()
      interpreter.run(inputBuffer, outputBuffer)

      Log.i(
        TAG,
        "Switches: Done calling interpreter.run(): %s (%.6f ms)".format(
          outputBuffer.array().contentToString(),
          (SystemClock.elapsedRealtimeNanos() - t0) / NANOS_IN_MILLIS
        )
      )
    }
  }

  private fun generateDummyAudioInput(inputBuffer: FloatBuffer) {
    val twoPiTimesFreq = 2 * Math.PI.toFloat() * 1000f
    for (i in 0 until modelInputLength) {
      val x = i.toFloat() / (modelInputLength - 1)
      inputBuffer.put(i, sin(twoPiTimesFreq * x.toDouble()).toFloat())
    }
  }

  /** Start recording and triggers recognition.  */
  @Synchronized
  private fun startAudioRecord() {
    if (isRecording) return
    setupAudioRecord()
    isClosed = false
    isRecording = true
  }

  /** Start a thread that runs model inference (i.e., recognition) at a regular interval.  */
  private fun startRecognition() {
    Log.i(TAG, "Starting Recognition")
    recognitionThread = RecognitionThread().apply {
      start()
    }
  }

  private fun setupAudioRecord() {
    var bufferSize = AudioRecord.getMinBufferSize(
      options.sampleRate,
      AudioFormat.CHANNEL_IN_MONO,
      AudioFormat.ENCODING_PCM_16BIT
    )
    Log.i(TAG, "min buffer size = $bufferSize")
    if (bufferSize == AudioRecord.ERROR || bufferSize == AudioRecord.ERROR_BAD_VALUE) {
      bufferSize = options.sampleRate * 2
      Log.w(TAG, "bufferSize has error or bad value")
    }
    // The buffer of AudioRecord should be larger than what model requires.
    val modelRequiredBufferSize = 2 * modelInputLength * Short.SIZE_BYTES
    if (bufferSize < modelRequiredBufferSize) {
      bufferSize = modelRequiredBufferSize
    }
    Log.i(TAG, "bufferSize = $bufferSize")
    audioRecord = AudioRecord(
      // including MIC, UNPROCESSED, and CAMCORDER.
      MediaRecorder.AudioSource.VOICE_RECOGNITION,
      options.sampleRate,
      AudioFormat.CHANNEL_IN_MONO,
      AudioFormat.ENCODING_PCM_16BIT,
      bufferSize
    )
    if (audioRecord.state != AudioRecord.STATE_INITIALIZED) {
      Log.e(TAG, "AudioRecord failed to initialize")
      return
    }
    Log.i(TAG, "Successfully initialized AudioRecord")

    audioRecord.startRecording()
    Log.i(TAG, "Successfully started AudioRecord recording")

    // Start recognition (model inference) thread.
    startRecognition()
  }

  private fun loadAudio(audioBuffer: ShortArray): Int {
    when (
      val loadedSamples = audioRecord.read(
        audioBuffer, 0, audioBuffer.size, AudioRecord.READ_NON_BLOCKING
      )
    ) {
      AudioRecord.ERROR_INVALID_OPERATION -> {
        Log.w(TAG, "AudioRecord.ERROR_INVALID_OPERATION")
      }
      AudioRecord.ERROR_BAD_VALUE -> {
        Log.w(TAG, "AudioRecord.ERROR_BAD_VALUE")
      }
      AudioRecord.ERROR_DEAD_OBJECT -> {
        Log.w(TAG, "AudioRecord.ERROR_DEAD_OBJECT")
      }
      AudioRecord.ERROR -> {
        Log.w(TAG, "AudioRecord.ERROR")
      }
      else -> {
        return loadedSamples
      }
    }
    // No new sample was loaded.
    return 0
  }

  private inner class RecognitionThread : Thread() {
    override fun run() {
      if (modelInputLength <= 0 || modelNumClasses <= 0) {
        Log.e(TAG, "Switches: Cannot start recognition because model is unavailable.")
        return
      }
      val outputBuffer = FloatBuffer.allocate(modelNumClasses)
      val recordingBuffer = ShortArray(modelInputLength)
      val circularBuffer = ShortArray(modelInputLength)
      var j = 0 // Indices for the circular buffer next write

      while (!isInterrupted) {

        // Wait for the next invocation
        try {
          TimeUnit.MILLISECONDS.sleep(recognitionPeriod)
        } catch (e: InterruptedException) {
          Log.w(TAG, "Sleep interrupted in recognition thread.")
        }

        // Load new audio samples
        val sampleCounts = loadAudio(recordingBuffer)
        if (sampleCounts == 0) {
          continue
        }

        // Copy new data into the circular buffer
        for (i in 0 until sampleCounts) {
          circularBuffer[j] = recordingBuffer[i]
          j = (j + 1) % circularBuffer.size
        }

        // Feed data to the input buffer.
        var samplesAreAllZero = true
        for (i in 0 until modelInputLength) {
          val s = if (i > options.pointsInAverage) {
            ((i - options.pointsInAverage + 1)..i).map {
              circularBuffer[(j + it) % modelInputLength]
            }
              .average()
          } else {
            circularBuffer[(i + j) % modelInputLength]
          }
          if (samplesAreAllZero && s.toInt() != 0) {
            samplesAreAllZero = false
          }
          inputBuffer.put(i, s.toFloat())
        }

        if (samplesAreAllZero) {
          Log.w(TAG, "No audio input: All audio samples are zero!")
          continue
        }
        val t0 = SystemClock.elapsedRealtimeNanos()
        inputBuffer.rewind()
        outputBuffer.rewind()
        interpreter.run(inputBuffer, outputBuffer)
        outputBuffer.rewind()
        outputBuffer.get(predictionProbs) // Copy data to predictionProbs.

        val probList = predictionProbs.map {
          if (it > probabilityThreshold) it else 0f
        }
        Log.i(TAG, "inference result: $probList")
        _probabilities.postValue(labelList.zip(probList).toMap())

        latestPredictionLatencyMs =
          ((SystemClock.elapsedRealtimeNanos() - t0) / 1e6).toFloat()
      }
    }
  }

  companion object {
    private const val TAG = "SoundClassifier"

    /** Number of nanoseconds in a millisecond  */
    private const val NANOS_IN_MILLIS = 1_000_000.toDouble()
  }
}

private fun String.toTitleCase() =
  splitToSequence("_")
    .map { it.capitalize(Locale.ROOT) }
    .joinToString(" ")
    .trim()
