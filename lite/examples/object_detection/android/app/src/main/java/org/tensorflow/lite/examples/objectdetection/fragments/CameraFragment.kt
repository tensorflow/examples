
/*
 * Copyright 2022 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *             http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.objectdetection.fragments

import android.annotation.SuppressLint
import android.content.res.Configuration
import android.graphics.Bitmap
import android.os.Bundle
import android.os.SystemClock
import android.speech.tts.TextToSpeech
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
import java.util.LinkedList
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import org.tensorflow.lite.examples.objectdetection.ObjectDetectorHelper
import org.tensorflow.lite.examples.objectdetection.R
import org.tensorflow.lite.examples.objectdetection.databinding.FragmentCameraBinding
import org.tensorflow.lite.task.vision.detector.Detection

class CameraFragment : Fragment(), ObjectDetectorHelper.DetectorListener {

    companion object {
        private const val TAG = "ObjectDetection"

        // A label must appear in this many consecutive analyzed frames.
        private const val REQUIRED_STABLE_FRAMES = 3

        // Minimum time between any two announcements.
        private const val MIN_ANNOUNCEMENT_GAP_MS = 2000L

        // Minimum delay before announcing the same label again.
        private const val SAME_LABEL_REPEAT_DELAY_MS = 6000L

        // Minimum confidence required before a detection can be spoken.
        private const val SPEECH_CONFIDENCE_THRESHOLD = 0.60f
    }

    private var _fragmentCameraBinding: FragmentCameraBinding? = null

    private val fragmentCameraBinding
        get() = _fragmentCameraBinding!!

    private lateinit var objectDetectorHelper: ObjectDetectorHelper
    private lateinit var bitmapBuffer: Bitmap
    private lateinit var cameraExecutor: ExecutorService

    private var preview: Preview? = null
    private var imageAnalyzer: ImageAnalysis? = null
    private var camera: Camera? = null
    private var cameraProvider: ProcessCameraProvider? = null

    // Text-to-Speech
    private var textToSpeech: TextToSpeech? = null
    private var ttsReady = false

    // Stable detection tracking
    private var candidateLabel = ""
    private var candidateCount = 0

    // Announcement tracking
    private var lastSpokenLabel = ""
    private var lastAnnouncementTime = 0L

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
        // Stop speech and release the TTS engine.
        textToSpeech?.stop()
        textToSpeech?.shutdown()
        textToSpeech = null
        ttsReady = false

        // Stop camera analysis.
        imageAnalyzer?.clearAnalyzer()
        cameraProvider?.unbindAll()

        imageAnalyzer = null
        preview = null
        camera = null
        cameraProvider = null

        // Release the background executor.
        if (::cameraExecutor.isInitialized) {
            cameraExecutor.shutdown()
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
    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        cameraExecutor = Executors.newSingleThreadExecutor()

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
            if (status == TextToSpeech.SUCCESS) {
                val result = textToSpeech?.setLanguage(Locale.US)

                ttsReady = result != null &&
                    result != TextToSpeech.LANG_MISSING_DATA &&
                    result != TextToSpeech.LANG_NOT_SUPPORTED

                Log.d(TAG, "Text-to-Speech ready: $ttsReady")
            } else {
                ttsReady = false
                Log.e(TAG, "Text-to-Speech initialization failed")
            }
        }
    }

    /**
     * Selects the highest-confidence label for the current frame.
     * The label must remain stable for several frames before it is spoken.
     */
    private fun announceDetection(results: MutableList<Detection>?) {
        if (!ttsReady || textToSpeech == null) return

        val bestCategory = results
            ?.asSequence()
            ?.mapNotNull { detection ->
                detection.categories.maxByOrNull { it.score }
            }
            ?.filter { category ->
                category.score >= SPEECH_CONFIDENCE_THRESHOLD &&
                    category.label.isNotBlank()
            }
            ?.maxByOrNull { it.score }

        // No sufficiently confident detection in this frame.
        if (bestCategory == null) {
            candidateLabel = ""
            candidateCount = 0
            return
        }

        val label = bestCategory.label

        // Require the same label in consecutive frames.
        if (label == candidateLabel) {
            candidateCount++
        } else {
            candidateLabel = label
            candidateCount = 1
        }

        if (candidateCount < REQUIRED_STABLE_FRAMES) {
            return
        }

        val now = SystemClock.elapsedRealtime()

        // Avoid speaking too frequently, even when different objects appear.
        if (now - lastAnnouncementTime < MIN_ANNOUNCEMENT_GAP_MS) {
            return
        }

        // Avoid repeating the same label too often.
        if (
            label == lastSpokenLabel &&
            now - lastAnnouncementTime < SAME_LABEL_REPEAT_DELAY_MS
        ) {
            return
        }

        val result = textToSpeech?.speak(
            "$label detected",
            TextToSpeech.QUEUE_FLUSH,
            null,
            "object_warning"
        )

        if (result == TextToSpeech.SUCCESS) {
            lastSpokenLabel = label
            lastAnnouncementTime = now

            // Require fresh stable frames before the next announcement.
            candidateLabel = ""
            candidateCount = 0

            Log.d(TAG, "Announced object: $label")
        }
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

                override fun onNothingSelected(parent: AdapterView<*>?) {
                    // No action required.
                }
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

                override fun onNothingSelected(parent: AdapterView<*>?) {
                    // No action required.
                }
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

        // Reset speech stability when detector settings change.
        candidateLabel = ""
        candidateCount = 0

        objectDetectorHelper.clearObjectDetector()
        binding.overlay.clear()
    }

    private fun setUpCamera() {
        val context = context ?: return

        val cameraProviderFuture =
            ProcessCameraProvider.getInstance(context)

        cameraProviderFuture.addListener(
            {
                if (_fragmentCameraBinding == null) return@addListener

                try {
                    cameraProvider = cameraProviderFuture.get()
                    bindCameraUseCases()
                } catch (exc: Exception) {
                    Log.e(TAG, "Camera initialization failed", exc)
                    onError("Unable to initialize the camera.")
                }
            },
            ContextCompat.getMainExecutor(context)
        )
    }

    @SuppressLint("UnsafeOptInUsageError")
    private fun bindCameraUseCases() {
        val provider = cameraProvider
            ?: run {
                onError("Camera provider is unavailable.")
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
                    } catch (exc: Exception) {
                        Log.e(TAG, "Object detection failed", exc)
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

            preview?.setSurfaceProvider(
                binding.viewFinder.surfaceProvider
            )
        } catch (exc: Exception) {
            Log.e(TAG, "Use case binding failed", exc)
            onError("Unable to start the camera preview.")
        }
    }

    private fun detectObjects(image: ImageProxy) {
        if (!::bitmapBuffer.isInitialized) return

        bitmapBuffer.copyPixelsFromBuffer(image.planes[0].buffer)

        val imageRotation = image.imageInfo.rotationDegrees

        objectDetectorHelper.detect(
            bitmapBuffer,
            imageRotation
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
                results ?: LinkedList<Detection>(),
                imageHeight,
                imageWidth
            )

            binding.overlay.invalidate()

            announceDetection(results)
        }
    }

    override fun onError(error: String) {
        activity?.runOnUiThread {
            context?.let { safeContext ->
                Toast.makeText(
                    safeContext,
                    error,
                    Toast.LENGTH_SHORT
                ).show()
            }
        }
    }
}
