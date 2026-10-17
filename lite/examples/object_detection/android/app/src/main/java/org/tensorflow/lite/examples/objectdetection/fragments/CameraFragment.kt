
/*
 * Copyright 2022 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0.
 * See https://www.apache.org/licenses/LICENSE-2.0
 */

package org.tensorflow.lite.examples.objectdetection.fragments

import android.annotation.SuppressLint
import android.content.res.Configuration
import android.graphics.Bitmap
import android.os.Bundle
import android.os.SystemClock
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.AdapterView
import android.widget.Toast
import androidx.camera.core.AspectRatio
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import androidx.navigation.Navigation
import org.tensorflow.lite.examples.objectdetection.ObjectDetectorHelper
import org.tensorflow.lite.examples.objectdetection.R
import org.tensorflow.lite.examples.objectdetection.databinding.FragmentCameraBinding
import org.tensorflow.lite.task.vision.detector.Detection
import java.io.DataOutputStream
import java.io.File
import java.io.FileInputStream
import java.net.InetSocketAddress
import java.net.Socket
import java.util.LinkedList
import java.util.Locale
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class CameraFragment :
    Fragment(),
    ObjectDetectorHelper.DetectorListener {

    companion object {
        private const val TAG = "ObjectDetection"

        // Raspberry Pi connection settings
        private const val PI_HOST = "10.242.215.42"
        private const val PI_PORT = 5001
        private const val CONNECT_TIMEOUT_MS = 3000
        private const val SOCKET_TIMEOUT_MS = 5000
        private const val MAX_AUDIO_SIZE = 10 * 1024 * 1024

        // Detection speech settings
        private const val REQUIRED_STABLE_FRAMES = 3
        private const val MIN_ANNOUNCEMENT_GAP_MS = 2000L
        private const val SAME_LABEL_REPEAT_DELAY_MS = 6000L
        private const val SPEECH_CONFIDENCE_THRESHOLD = 0.60f
    }

    private var _fragmentCameraBinding: FragmentCameraBinding? = null

    private val fragmentCameraBinding
        get() = _fragmentCameraBinding!!

    private lateinit var objectDetectorHelper: ObjectDetectorHelper
    private lateinit var bitmapBuffer: Bitmap
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var audioExecutor: ExecutorService

    private var preview: Preview? = null
    private var imageAnalyzer: ImageAnalysis? = null
    private var camera: Camera? = null
    private var cameraProvider: ProcessCameraProvider? = null

    // Text-to-Speech
    private var textToSpeech: TextToSpeech? = null
    private var ttsReady = false
    private var speechPending = false

    // Synthesized audio files, indexed by utterance ID
    private val audioFiles = ConcurrentHashMap<String, File>()

    // Stable detection tracking
    private var candidateLabel = ""
    private var candidateCount = 0

    // Announcement tracking
    private var lastSpokenLabel = ""
    private var lastAnnouncementTime = 0L
    private var utteranceCounter = 0

    override fun onResume() {
        super.onResume()

        if (!PermissionsFragment.hasPermissions(requireContext())) {
            Navigation.findNavController(
                requireActivity(),
                R.id.fragment_container
            ).navigate(
                CameraFragmentDirections.actionCameraToPermissions()
            )
        }
    }

    override fun onDestroyView() {
        // Stop camera analysis before releasing the view.
        imageAnalyzer?.clearAnalyzer()
        cameraProvider?.unbindAll()

        imageAnalyzer = null
        preview = null
        camera = null
        cameraProvider = null

        // Stop TTS and release the engine.
        textToSpeech?.stop()
        textToSpeech?.shutdown()
        textToSpeech = null
        ttsReady = false
        speechPending = false

        // Release generated files.
        audioFiles.values.forEach { file ->
            runCatching { file.delete() }
        }
        audioFiles.clear()

        if (::cameraExecutor.isInitialized) {
            cameraExecutor.shutdown()
        }

        if (::audioExecutor.isInitialized) {
            audioExecutor.shutdownNow()
        }

        _fragmentCameraBinding = null
        super.onDestroyView()
    }

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _fragmentCameraBinding =
            FragmentCameraBinding.inflate(inflater, container, false)

        return fragmentCameraBinding.root
    }

    @SuppressLint("MissingPermission")
    override fun onViewCreated(
        view: View,
        savedInstanceState: Bundle?
    ) {
        super.onViewCreated(view, savedInstanceState)

        cameraExecutor = Executors.newSingleThreadExecutor()
        audioExecutor = Executors.newSingleThreadExecutor()

        initializeTextToSpeech()

        objectDetectorHelper = ObjectDetectorHelper(
            context = requireContext(),
            objectDetectorListener = this
        )

        fragmentCameraBinding.viewFinder.post {
            if (_fragmentCameraBinding != null) {
                setUpCamera()
            }
        }

        initBottomSheetControls()
        updateControlsUi()
    }

    private fun initializeTextToSpeech() {
        ttsReady = false

        textToSpeech = TextToSpeech(requireContext()) { status ->
            if (status != TextToSpeech.SUCCESS) {
                Log.e(TAG, "TTS initialization failed")
                showToast("Text-to-Speech initialization failed.")
                return@TextToSpeech
            }

            val engine = textToSpeech ?: return@TextToSpeech
            val languageResult = engine.setLanguage(Locale.US)

            ttsReady =
                languageResult != TextToSpeech.LANG_MISSING_DATA &&
                languageResult != TextToSpeech.LANG_NOT_SUPPORTED

            if (!ttsReady) {
                Log.e(TAG, "TTS language is unavailable")
                showToast("English speech is unavailable.")
                return@TextToSpeech
            }

            engine.setOnUtteranceProgressListener(
                object : UtteranceProgressListener() {

                    override fun onStart(utteranceId: String?) {
                        Log.d(TAG, "Speech synthesis started: $utteranceId")
                    }

                    override fun onDone(utteranceId: String?) {
                        if (utteranceId == null) return

                        val file = audioFiles[utteranceId] ?: return

                        if (!::audioExecutor.isInitialized ||
                            audioExecutor.isShutdown
                        ) {
                            audioFiles.remove(utteranceId)
                            file.delete()
                            return
                        }

                        try {
                            audioExecutor.execute {
                                try {
                                    sendAudioToPi(file)
                                } catch (e: Exception) {
                                    Log.e(
                                        TAG,
                                        "Failed to transmit synthesized audio",
                                        e
                                    )
                                    showToast(
                                        "Could not send audio to Raspberry Pi."
                                    )
                                } finally {
                                    audioFiles.remove(utteranceId)
                                    file.delete()

                                    activity?.runOnUiThread {
                                        speechPending = false
                                    }
                                }
                            }
                        } catch (e: Exception) {
                            audioFiles.remove(utteranceId)
                            file.delete()
                            Log.e(TAG, "Audio executor unavailable", e)
                            activity?.runOnUiThread {
                                speechPending = false
                            }
                        }
                    }

                    @Deprecated("Deprecated in Android API")
                    override fun onError(utteranceId: String?) {
                        handleSynthesisError(utteranceId)
                    }

                    override fun onError(
                        utteranceId: String?,
                        errorCode: Int
                    ) {
                        Log.e(
                            TAG,
                            "Speech synthesis error: $errorCode"
                        )
                        handleSynthesisError(utteranceId)
                    }
                }
            )

            Log.d(TAG, "TTS ready: $ttsReady")
        }
    }

    private fun handleSynthesisError(utteranceId: String?) {
        if (utteranceId != null) {
            audioFiles.remove(utteranceId)?.delete()
        }

        activity?.runOnUiThread {
            speechPending = false
        }
    }

    /**
     * Generate a WAV file locally, then send it to the Raspberry Pi.
     * This function should be called on the main thread.
     */
    private fun synthesizeAndSend(text: String) {
        val engine = textToSpeech

        if (!ttsReady || engine == null || speechPending) {
            return
        }

        if (!::audioExecutor.isInitialized || audioExecutor.isShutdown) {
            return
        }

        val utteranceId = "blind_assist_${++utteranceCounter}"
        val outputFile = File(
            requireContext().cacheDir,
            "$utteranceId.wav"
        )

        speechPending = true
        audioFiles[utteranceId] = outputFile

        val result = engine.synthesizeToFile(
            text,
            Bundle(),
            outputFile,
            utteranceId
        )

        if (result != TextToSpeech.SUCCESS) {
            audioFiles.remove(utteranceId)
            outputFile.delete()
            speechPending = false
            Log.e(TAG, "Could not start WAV synthesis")
        }
    }

    /**
     * Protocol expected by audio_receiver.py:
     * 4-byte big-endian WAV length, followed by WAV bytes.
     *
     * Network work runs on audioExecutor, never on the UI thread.
     */
    private fun sendAudioToPi(file: File) {
        if (!file.exists()) {
            throw IllegalStateException("Synthesized WAV file is missing")
        }

        val audioLength = file.length()

        if (audioLength <= 0 || audioLength > MAX_AUDIO_SIZE) {
            throw IllegalStateException(
                "Invalid audio file size: $audioLength"
            )
        }

        Socket().use { socket ->
            socket.connect(
                InetSocketAddress(PI_HOST, PI_PORT),
                CONNECT_TIMEOUT_MS
            )
            socket.soTimeout = SOCKET_TIMEOUT_MS
            socket.tcpNoDelay = true

            DataOutputStream(socket.getOutputStream()).use { output ->
                // DataOutputStream writes integers in big-endian order.
                output.writeInt(audioLength.toInt())

                FileInputStream(file).use { input ->
                    val buffer = ByteArray(8192)
                    var totalSent = 0L

                    while (true) {
                        val count = input.read(buffer)
                        if (count == -1) break

                        output.write(buffer, 0, count)
                        totalSent += count
                    }

                    output.flush()

                    if (totalSent != audioLength) {
                        throw IllegalStateException(
                            "Audio transmission was incomplete"
                        )
                    }
                }
            }
        }

        Log.i(TAG, "Audio sent to Raspberry Pi ($audioLength bytes)")
    }

    /**
     * Select the highest-confidence label, require stable frames,
     * then synthesize speech and transmit it to the Pi.
     */
    private fun announceDetection(results: MutableList<Detection>?) {
        if (!ttsReady || speechPending) return

        val bestCategory = results
            ?.asSequence()
            ?.mapNotNull { detection ->
                detection.categories.maxByOrNull { it.score }
            }
            ?.filter {
                it.score >= SPEECH_CONFIDENCE_THRESHOLD &&
                    it.label.isNotBlank()
            }
            ?.maxByOrNull { it.score }

        if (bestCategory == null) {
            candidateLabel = ""
            candidateCount = 0
            return
        }

        val label = bestCategory.label

        if (label == candidateLabel) {
            candidateCount++
        } else {
            candidateLabel = label
            candidateCount = 1
        }

        if (candidateCount < REQUIRED_STABLE_FRAMES) return

        val now = SystemClock.elapsedRealtime()

        if (now - lastAnnouncementTime < MIN_ANNOUNCEMENT_GAP_MS) {
            return
        }

        if (
            label == lastSpokenLabel &&
            now - lastAnnouncementTime < SAME_LABEL_REPEAT_DELAY_MS
        ) {
            return
        }

        // Reserve the speech slot before starting synthesis.
        lastSpokenLabel = label
        lastAnnouncementTime = now
        candidateLabel = ""
        candidateCount = 0

        Log.i(TAG, "Announcing detected object: $label")
        synthesizeAndSend("$label detected")
    }

    private fun initBottomSheetControls() {
        fragmentCameraBinding.bottomSheetLayout.thresholdMinus
            .setOnClickListener {
                if (objectDetectorHelper.threshold > 0.1f) {
                    objectDetectorHelper.threshold =
                        (objectDetectorHelper.threshold - 0.1f)
                            .coerceAtLeast(0.1f)
                    updateControlsUi()
                }
            }

        fragmentCameraBinding.bottomSheetLayout.thresholdPlus
            .setOnClickListener {
                if (objectDetectorHelper.threshold < 0.9f) {
                    objectDetectorHelper.threshold =
                        (objectDetectorHelper.threshold + 0.1f)
                            .coerceAtMost(0.9f)
                    updateControlsUi()
                }
            }

        fragmentCameraBinding.bottomSheetLayout.maxResultsMinus
            .setOnClickListener {
                if (objectDetectorHelper.maxResults > 1) {
                    objectDetectorHelper.maxResults--
                    updateControlsUi()
                }
            }

        fragmentCameraBinding.bottomSheetLayout.maxResultsPlus
            .setOnClickListener {
                if (objectDetectorHelper.maxResults < 5) {
                    objectDetectorHelper.maxResults++
                    updateControlsUi()
                }
            }

        fragmentCameraBinding.bottomSheetLayout.threadsMinus
            .setOnClickListener {
                if (objectDetectorHelper.numThreads > 1) {
                    objectDetectorHelper.numThreads--
                    updateControlsUi()
                }
            }

        fragmentCameraBinding.bottomSheetLayout.threadsPlus
            .setOnClickListener {
                if (objectDetectorHelper.numThreads < 4) {
                    objectDetectorHelper.numThreads++
                    updateControlsUi()
                }
            }

        fragmentCameraBinding.bottomSheetLayout.spinnerDelegate
            .setSelection(0, false)

        fragmentCameraBinding.bottomSheetLayout.spinnerDelegate
            .onItemSelectedListener =
            object : AdapterView.OnItemSelectedListener {

                override fun onItemSelected(
                    parent: AdapterView<*>?,
                    view: View?,
                    position: Int,
                    id: Long
                ) {
                    if (objectDetectorHelper.currentDelegate != position) {
                        objectDetectorHelper.currentDelegate = position
                        updateControlsUi()
                    }
                }

                override fun onNothingSelected(parent: AdapterView<*>?) = Unit
            }

        fragmentCameraBinding.bottomSheetLayout.spinnerModel
            .setSelection(0, false)

        fragmentCameraBinding.bottomSheetLayout.spinnerModel
            .onItemSelectedListener =
            object : AdapterView.OnItemSelectedListener {

                override fun onItemSelected(
                    parent: AdapterView<*>?,
                    view: View?,
                    position: Int,
                    id: Long
                ) {
                    if (objectDetectorHelper.currentModel != position) {
                        objectDetectorHelper.currentModel = position
                        updateControlsUi()
                    }
                }

                override fun onNothingSelected(parent: AdapterView<*>?) = Unit
            }
    }

    private fun updateControlsUi() {
        val binding = _fragmentCameraBinding ?: return

        binding.bottomSheetLayout.maxResultsValue.text =
            objectDetectorHelper.maxResults.toString()

        binding.bottomSheetLayout.thresholdValue.text =
            String.format(
                Locale.US,
                "%.2f",
                objectDetectorHelper.threshold
            )

        binding.bottomSheetLayout.threadsValue.text =
            objectDetectorHelper.numThreads.toString()

        candidateLabel = ""
        candidateCount = 0

        objectDetectorHelper.clearObjectDetector()
        binding.overlay.clear()
    }

    private fun setUpCamera() {
        val safeContext = context ?: return

        val cameraProviderFuture =
            ProcessCameraProvider.getInstance(safeContext)

        cameraProviderFuture.addListener(
            {
                if (_fragmentCameraBinding == null) return@addListener

                try {
                    cameraProvider = cameraProviderFuture.get()
                    bindCameraUseCases()
                } catch (e: Exception) {
                    Log.e(TAG, "Camera initialization failed", e)
                    showToast("Unable to initialize the camera.")
                }
            },
            ContextCompat.getMainExecutor(safeContext)
        )
    }

    @SuppressLint("UnsafeOptInUsageError")
    private fun bindCameraUseCases() {
        val provider = cameraProvider ?: run {
            showToast("Camera provider is unavailable.")
            return
        }

        val binding = _fragmentCameraBinding ?: return

        val cameraSelector = CameraSelector.Builder()
            .requireLensFacing(CameraSelector.LENS_FACING_BACK)
            .build()

        preview = Preview.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_4_3)
            .setTargetRotation(binding.viewFinder.display.rotation)
            .build()

        imageAnalyzer = ImageAnalysis.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_4_3)
            .setTargetRotation(binding.viewFinder.display.rotation)
            .setBackpressureStrategy(
                ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST
            )
            .setOutputImageFormat(OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build()
            .also { analyzer ->
                analyzer.setAnalyzer(cameraExecutor) { image ->
                    try {
                        if (_fragmentCameraBinding == null) {
                            return@setAnalyzer
                        }

                        if (
                            !::bitmapBuffer.isInitialized ||
                            bitmapBuffer.width != image.width ||
                            bitmapBuffer.height != image.height
                        ) {
                            bitmapBuffer = Bitmap.createBitmap(
                                image.width,
                                image.height,
                                Bitmap.Config.ARGB_8888
                            )
                        }

                        detectObjects(image)
                    } catch (e: Exception) {
                        Log.e(TAG, "Object detection failed", e)
                    } finally {
                        image.close()
                    }
                }
            }

        provider.unbindAll()

        try {
            camera = provider.bindToLifecycle(
                this,
                cameraSelector,
                preview,
                imageAnalyzer
            )

            preview?.setSurfaceProvider(binding.viewFinder.surfaceProvider)
        } catch (e: Exception) {
            Log.e(TAG, "Use case binding failed", e)
            showToast("Unable to start the camera preview.")
        }
    }

    private fun detectObjects(image: ImageProxy) {
        if (!::bitmapBuffer.isInitialized) return

        bitmapBuffer.copyPixelsFromBuffer(image.planes[0].buffer)

        objectDetectorHelper.detect(
            bitmapBuffer,
            image.imageInfo.rotationDegrees
        )
    }

    override fun onConfigurationChanged(newConfig: Configuration) {
        super.onConfigurationChanged(newConfig)

        _fragmentCameraBinding?.let { binding ->
            imageAnalyzer?.targetRotation =
                binding.viewFinder.display.rotation
        }
    }

    override fun onResults(
        results: MutableList<Detection>?,
        inferenceTime: Long,
        imageHeight: Int,
        imageWidth: Int
    ) {
        activity?.runOnUiThread {
            val binding = _fragmentCameraBinding
                ?: return@runOnUiThread

            binding.bottomSheetLayout.inferenceTimeVal.text =
                String.format(Locale.US, "%d ms", inferenceTime)

            binding.overlay.setResults(
                results ?: LinkedList(),
                imageHeight,
                imageWidth
            )

            binding.overlay.invalidate()
            announceDetection(results)
        }
    }

    override fun onError(error: String) {
        showToast(error)
    }

    private fun showToast(message: String) {
        activity?.runOnUiThread {
            context?.let { safeContext ->
                Toast.makeText(
                    safeContext,
                    message,
                    Toast.LENGTH_SHORT
                ).show()
            }
        }
    }
}
