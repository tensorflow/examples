/*
 * Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
import androidx.lifecycle.DefaultLifecycleObserver
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import java.io.BufferedReader
import java.io.IOException
import java.io.InputStreamReader
import java.nio.FloatBuffer
import java.util.*
import java.util.concurrent.TimeUnit
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock
import kotlin.math.ceil
import kotlin.math.sin

class SoundClassifier(context: Context) : DefaultLifecycleObserver {
    val isRecording: Boolean
        get() = recordingThread?.isAlive == true

    /** As the result of sound classification, this value emits map of probabilities */
    val probabilities: LiveData<Map<String, Float>>
        get() = _probabilities
    private val _probabilities = MutableLiveData<Map<String, Float>>()

    private val recordingBufferLock: ReentrantLock = ReentrantLock()

    var isClosed: Boolean = true
        private set

    var lifecycleOwner: LifecycleOwner? = null
        set(value) {
            field = value?.also {
                it.lifecycle.addObserver(this)
            }
        }

    /** Paused by user */
    var isPaused: Boolean = false
        set(value) {
            field = value
            if (value) stop() else start()
        }

    var overlapFactor: Float = DEFAULT_OVERLAP_FACTOR
        set(value) {
            field = value.also {
                recognitionPeriod = (1000L * (1 - value)).toLong()
            }
        }

    /** Names of the model's output classes.  */
    lateinit var labelList: List<String>
        private set

    /** How many milliseconds between consecutive model inference calls.  */
    private var recognitionPeriod = (1000L * (1 - DEFAULT_OVERLAP_FACTOR)).toLong()

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

    private var recordingThread: Thread? = null
    private var recognitionThread: Thread? = null

    private var recordingOffset = 0
    private lateinit var recordingBuffer: ShortArray

    /** Buffer that holds audio PCM sample that are fed to the TFLite model for inference.  */
    private lateinit var inputBuffer: FloatBuffer

    init {
        loadWordLabels(context)
        setupInterpreter(context)
        warmUpModel()
    }

    override fun onResume(owner: LifecycleOwner) = start()

    override fun onPause(owner: LifecycleOwner) = stop()

    override fun onDestroy(owner: LifecycleOwner) = close()

    fun start() {
        if (!isPaused) {
            startAudioRecord()
        }
    }

    fun stop() {
        if (isClosed || !isRecording) {
            return
        }
        recordingThread?.interrupt()
        recognitionThread?.interrupt()

        _probabilities.postValue(labelList.associateWith { 0f })
    }

    fun close() {
        stop()

        if (isClosed) {
            return
        }
        interpreter.close()

        isClosed = true
    }

    // Retrieve word labels from "labels.txt" file
    private fun loadWordLabels(context: Context) {
        try {
            val reader = BufferedReader(InputStreamReader(context.assets.open(METADATA_PATH)))
            val wordList = mutableListOf<String>()
            reader.useLines { lines ->
                lines.forEach {
                    wordList.add(it.split(" ").last())
                }
            }
            labelList = wordList.map { it.toTitleCase() }
        } catch (e: IOException) {
            Log.e(TAG, "Failed to read model metadata.json: ${e.message}")
        }
    }

    private fun setupInterpreter(context: Context) {
        interpreter = try {
            // Model path relative to the assets folder.
            val tfliteBuffer =
                FileUtil.loadMappedFile(context, MODEL_PATH)
            Log.i(TAG, "Done creating TFLite buffer from $MODEL_PATH")
            Interpreter(
                tfliteBuffer,
                Interpreter.Options()
            )
        } catch (e: IOException) {
            Log.e(TAG, "Switches: Failed to call TFLite model(): ${e.message}")
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
                "Mismatch between metadata number of classes (${labelList.size})" + " and model output length (${modelNumClasses})"
            )
        }
        // Fill the array with NaNs initially.
        predictionProbs = FloatArray(modelNumClasses) { Float.NaN }

        inputBuffer = FloatBuffer.allocate(modelInputLength)
    }

    private fun warmUpModel() {
        generateDummyAudioInput(inputBuffer)
        for (n in 0 until NUM_WARMUP_RUNS) {
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

    /** Start a thread to pull audio samples in continuously.  */
    @Synchronized
    private fun startAudioRecord() {
        if (isRecording) {
            return
        }
        recordingThread = AudioRecordingThread().apply {
            start()
        }
        isClosed = false
    }

    /** Start a thread that runs model inference (i.e., recognition) at a regular interval.  */
    private fun startRecognition() {
        recognitionThread = RecognitionThread().apply {
            start()
        }
    }

    /** Runnable class to run a thread for audio recording */
    private inner class AudioRecordingThread : Thread() {
        override fun run() {
            var bufferSize = AudioRecord.getMinBufferSize(
                SAMPLE_RATE_HZ,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT
            )
            if (bufferSize == AudioRecord.ERROR || bufferSize == AudioRecord.ERROR_BAD_VALUE) {
                bufferSize = SAMPLE_RATE_HZ * 2
                Log.w(TAG, "bufferSize has error or bad value")
            }
            Log.i(TAG, "bufferSize = $bufferSize")
            val record = AudioRecord(
                // including MIC, UNPROCESSED, and CAMCORDER.
                MediaRecorder.AudioSource.VOICE_RECOGNITION,
                SAMPLE_RATE_HZ,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
                bufferSize
            )
            if (record.state != AudioRecord.STATE_INITIALIZED) {
                Log.e(TAG, "AudioRecord failed to initialize")
                return
            }
            Log.i(TAG, "Successfully initialized AudioRecord")
            val bufferSamples = bufferSize / 2
            val audioBuffer = ShortArray(bufferSamples)
            val recordingBufferSamples =
                ceil(modelInputLength.toFloat() / bufferSamples.toDouble())
                    .toInt() * bufferSamples
            Log.i(TAG, "recordingBufferSamples = $recordingBufferSamples")
            recordingOffset = 0
            recordingBuffer = ShortArray(recordingBufferSamples)
            record.startRecording()
            Log.i(TAG, "Successfully started AudioRecord recording")

            // Start recognition (model inference) thread.
            startRecognition()

            while (!isInterrupted) {
                try {
                    TimeUnit.MILLISECONDS.sleep(AUDIO_PULL_PERIOD_MS)
                } catch (e: InterruptedException) {
                    Log.w(TAG, "Sleep interrupted in audio recording thread.")
                    break
                }
                when (record.read(audioBuffer, 0, audioBuffer.size)) {
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
                    bufferSamples -> {
                        // We apply locks here to avoid two separate threads (the recording and
                        // recognition threads) reading and writing from the recordingBuffer at the same
                        // time, which can cause the recognition thread to read garbled audio snippets.
                        recordingBufferLock.withLock {
                            audioBuffer.copyInto(
                                recordingBuffer,
                                recordingOffset,
                                0,
                                bufferSamples
                            )
                            recordingOffset =
                                (recordingOffset + bufferSamples) % recordingBufferSamples
                        }
                    }
                }
            }
        }
    }

    private inner class RecognitionThread : Thread() {
        override fun run() {
            if (modelInputLength <= 0 || modelNumClasses <= 0) {
                Log.e(TAG, "Switches: Cannot start recognition because model is unavailable.")
                return
            }
            val outputBuffer = FloatBuffer.allocate(modelNumClasses)
            while (!isInterrupted) {
                try {
                    TimeUnit.MILLISECONDS.sleep(recognitionPeriod)
                } catch (e: InterruptedException) {
                    Log.w(TAG, "Sleep interrupted in recognition thread.")
                    break
                }
                var samplesAreAllZero = true

                recordingBufferLock.withLock {
                    var j = (recordingOffset - modelInputLength) % modelInputLength
                    if (j < 0) {
                        j += modelInputLength
                    }
                    //Log.i(">>>", "recordingOffset: $recordingOffset, modelInputLength: $modelInputLength, j: $j")

                    for (i in 0 until modelInputLength) {
                        val s = if (i >= POINTS_IN_AVG && j >= POINTS_IN_AVG) {
                            ((j - POINTS_IN_AVG + 1)..j).map { recordingBuffer[it % modelInputLength] }
                                .average()
                        } else {
                            recordingBuffer[j % modelInputLength]
                        }
                        j += 1

                        if (samplesAreAllZero && s.toInt() != 0) {
                            samplesAreAllZero = false
                        }
                        // TODO(cais): Explore better way of reading float samples directly from the
                        // AudioSource and using bulk put() instead of sample-by-sample put() here.
                        inputBuffer.put(i, s.toFloat())
                    }
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
                    if (it > PROB_THRESHOLD) it else 0f
                }
                _probabilities.postValue(labelList.zip(probList).toMap())

                latestPredictionLatencyMs =
                    ((SystemClock.elapsedRealtimeNanos() - t0) / 1e6).toFloat()
            }
        }
    }

    companion object {
        const val DEFAULT_OVERLAP_FACTOR = 0.8.toFloat()

        private const val TAG = "SoundClassifier"

        /** Path of the converted model label file, relative to the assets/ directory.  */
        private const val METADATA_PATH = "labels.txt"

        /** Path of the converted .tflite file, relative to the assets/ directory.  */
        private const val MODEL_PATH = "combined_model.tflite"

        /** Number of nanoseconds in a millisecond  */
        private const val NANOS_IN_MILLIS = 1_000_000.toDouble()

        /** Hard code the required audio sample rate in Hz.  */
        private const val SAMPLE_RATE_HZ = 44_100

        /** How many milliseconds to sleep between successive audio sample pulls.  */
        private const val AUDIO_PULL_PERIOD_MS = 50L

        /** Number of warm up runs to do after loading the TFLite model.  */
        private const val NUM_WARMUP_RUNS = 3

        /** Probability value above which a class is labeled as active (i.e., detected) the display.  */
        private const val PROB_THRESHOLD = 0.3f

        private const val POINTS_IN_AVG = 10
    }
}

private fun String.toTitleCase() =
    splitToSequence("_")
        .map { it.capitalize(Locale.ROOT) }
        .joinToString(" ")
        .trim()