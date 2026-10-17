
/*
 * Copyright 2022 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0.
 */

package org.tensorflow.lite.examples.objectdetection.fragments

import android.annotation.SuppressLint
import android.content.res.Configuration
import android.graphics.Bitmap
import android.os.Bundle
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

    private val TAG = "ObjectDetection"

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
    private var lastSpokenLabel = ""
    private var lastSpokenTime = 0L

    override fun onResume() {
        super.onResume()

        if (!PermissionsFragment.hasPermissions(requireContext())) {
            Navigation.findNavController(
                requireActivity(),
                R.id.fragment_container
            ).navigate(CameraFragmentDirections.actionCameraToPermissions())
        }
    }

    override fun onDestroyView() {
        textToSpeech?.stop()
        textToSpeech?.shutdown()
        textToSpeech = null
        ttsReady = false

        cameraProvider?.unbindAll()
        cameraProvider = null

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

        initializeTextToSpeech()

        objectDetectorHelper = ObjectDetectorHelper(
            context = requireContext(),
            objectDetectorListener = this
        )

        // Initialize the camera background executor.
        cameraExecutor = Executors.newSingleThreadExecutor()

        fragmentCameraBinding.viewFinder.post {
            setUpCamera()
        }

        initBottomSheetControls()
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
                Log.e(TAG, "Text-to-Speech initialization failed")
            }
        }
    }

    private fun announceDetection(results: MutableList<Detection>?) {
        if (!ttsReady || results.isNullOrEmpty()) return

        val bestCategory = results
            .asSequence()
            .mapNotNull { detection ->
                detection.categories.maxByOrNull { it.score }
            }
            .filter { it.score >= 0.60f }
            .maxByOrNull { it.score }
            ?: return

        val label = bestCategory.label
        if (label.isBlank()) return

        val now = System.currentTimeMillis()

        // Announce a new label immediately.
        // Repeat the same label at most once every three seconds.
        if (
            label == lastSpokenLabel &&
            now - lastSpokenTime < 3000L
        ) {
            return
        }

        val accepted = textToSpeech?.speak(
            "$label ahead",
            TextToSpeech.QUEUE_FLUSH,
            null,
            "object_warning"
        )

        if (accepted == TextToSpeech.SUCCESS) {
            lastSpokenLabel = label
            lastSpokenTime = now
        }
    }

    private fun initBottomSheetControls() {
        fragmentCameraBinding.bottomSheetLayout.thresholdMinus
            .setOnClickListener {
                if (objectDetectorHelper.threshold >= 0.1f) {
                    objectDetectorHelper.threshold -= 0.1f
                    updateControlsUi()
                }
            }

        fragmentCameraBinding.bottomSheetLayout.thresholdPlus
            .setOnClickListener {
                if (objectDetectorHelper.threshold <= 0.8f) {
                    objectDetectorHelper.threshold += 0.1f
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
                    objectDetectorHelper.currentDelegate = position
                    updateControlsUi()
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
                    objectDetectorHelper.currentModel = position
                    updateControlsUi()
                }

                override fun onNothingSelected(parent: AdapterView<*>?) {
                    // No action required.
                }
            }
    }

    private fun updateControlsUi() {
        fragmentCameraBinding.bottomSheetLayout.maxResultsValue.text =
            objectDetectorHelper.maxResults.toString()

        fragmentCameraBinding.bottomSheetLayout.thresholdValue.text =
            String.format("%.2f", objectDetectorHelper.threshold)

        fragmentCameraBinding.bottomSheetLayout.threadsValue.text =
            objectDetectorHelper.numThreads.toString()

        objectDetectorHelper.clearObjectDetector()
        fragmentCameraBinding.overlay.clear()
    }

    private fun setUpCamera() {
        val cameraProviderFuture =
            ProcessCameraProvider.getInstance(requireContext())

        cameraProviderFuture.addListener(
            {
                cameraProvider = cameraProviderFuture.get()
                bindCameraUseCases()
            },
            ContextCompat.getMainExecutor(requireContext())
        )
    }

    @SuppressLint("UnsafeOptInUsageError")
    private fun bindCameraUseCases() {
        val provider = cameraProvider
            ?: throw IllegalStateException("Camera initialization failed.")

        val cameraSelector = CameraSelector.Builder()
            .requireLensFacing(CameraSelector.LENS_FACING_BACK)
            .build()

        preview = Preview.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_4_3)
            .setTargetRotation(fragmentCameraBinding.viewFinder.display.rotation)
            .build()

        imageAnalyzer = ImageAnalysis.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_4_3)
            .setTargetRotation(fragmentCameraBinding.viewFinder.display.rotation)
            .setBackpressureStrategy(
                ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST
            )
            .setOutputImageFormat(OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build()
            .also { analyzer ->
                analyzer.setAnalyzer(cameraExecutor) { image ->
                    try {
                        if (!::bitmapBuffer.isInitialized) {
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
                fragmentCameraBinding.viewFinder.surfaceProvider
            )
        } catch (exc: Exception) {
            Log.e(TAG, "Use case binding failed", exc)
        }
    }

    private fun detectObjects(image: ImageProxy) {
        bitmapBuffer.copyPixelsFromBuffer(image.planes[0].buffer)

        val imageRotation = image.imageInfo.rotationDegrees
        objectDetectorHelper.detect(bitmapBuffer, imageRotation)
    }

    override fun onConfigurationChanged(newConfig: Configuration) {
        super.onConfigurationChanged(newConfig)
        imageAnalyzer?.targetRotation =
            fragmentCameraBinding.viewFinder.display.rotation
    }

    override fun onResults(
        results: MutableList<Detection>?,
        inferenceTime: Long,
        imageHeight: Int,
        imageWidth: Int
    ) {
        activity?.runOnUiThread {
            val binding = _fragmentCameraBinding ?: return@runOnUiThread

            binding.bottomSheetLayout.inferenceTimeVal.text =
                String.format("%d ms", inferenceTime)

            binding.overlay.setResults(
                results ?: LinkedList<Detection>(),
                imageHeight,
                imageWidth
            )

            binding.overlay.invalidate()

            // Speak the best detection above the confidence threshold.
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
