/*
 * Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

package org.tensorflow.lite.examples.videoclassification

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.hardware.camera2.CaptureRequest
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import android.util.Range
import android.view.View
import android.view.ViewTreeObserver.OnGlobalLayoutListener
import android.view.WindowManager
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.LinearLayout
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.camera2.interop.Camera2Interop
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.google.android.material.bottomsheet.BottomSheetBehavior
import org.tensorflow.lite.examples.videoclassification.databinding.ActivityMainBinding
import org.tensorflow.lite.examples.videoclassification.ml.VideoClassifier
import org.tensorflow.lite.support.label.Category
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

@androidx.camera.core.ExperimentalGetImage
@androidx.camera.camera2.interop.ExperimentalCamera2Interop
class MainActivity : AppCompatActivity() {
    companion object {
        private const val REQUEST_CODE_PERMISSIONS = 10
        private const val TAG = "TFLite-VidClassify"
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
        private const val MAX_RESULT = 3
        private const val MODEL_MOVINET_A0_FILE = "movinet_a0_stream_int8.tflite"
        private const val MODEL_MOVINET_A1_FILE = "movinet_a1_stream_int8.tflite"
        private const val MODEL_MOVINET_A2_FILE = "movinet_a2_stream_int8.tflite"
        private const val MODEL_LABEL_FILE = "kinetics600_label_map.txt"
        private const val MODEL_FPS = 5 // Ensure the input images are fed to the model at this fps.
        private const val MODEL_FPS_ERROR_RANGE = 0.1 // Acceptable error range in fps.
        private const val MAX_CAPTURE_FPS = 20
    }

    private val lock = Any()
    private lateinit var binding: ActivityMainBinding
    private lateinit var executor: ExecutorService
    private lateinit var sheetBehavior: BottomSheetBehavior<LinearLayout>

    private var videoClassifier: VideoClassifier? = null
    private var numThread = 1
    private var modelPosition = 0
    private var lastInferenceStartTime: Long = 0
    private var changeModelListener = object : AdapterView.OnItemSelectedListener {
        override fun onNothingSelected(parent: AdapterView<*>?) {
            // do nothing
        }

        override fun onItemSelected(
            parent: AdapterView<*>?,
            view: View?,
            position: Int,
            id: Long
        ) {
            modelPosition = position
            createClassifier()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        // Initialize the view layout.
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // Initialize the bottom sheet.
        sheetBehavior = BottomSheetBehavior.from(binding.bottomSheet.bottomSheetLayout)
        binding.bottomSheet.gestureLayout.viewTreeObserver.addOnGlobalLayoutListener(object :
            OnGlobalLayoutListener {
            override fun onGlobalLayout() {
                binding.bottomSheet.gestureLayout.viewTreeObserver.removeOnGlobalLayoutListener(this)
                val height = binding.bottomSheet.gestureLayout.measuredHeight
                sheetBehavior.peekHeight = height
            }
        })
        sheetBehavior.isHideable = false
        sheetBehavior.addBottomSheetCallback(object : BottomSheetBehavior.BottomSheetCallback() {
            override fun onStateChanged(bottomSheet: View, newState: Int) {
                when (newState) {
                    BottomSheetBehavior.STATE_EXPANDED -> {
                        binding.bottomSheet.bottomSheetArrow.setImageResource(R.drawable.icn_chevron_down)
                    }
                    BottomSheetBehavior.STATE_COLLAPSED, BottomSheetBehavior.STATE_SETTLING -> {
                        binding.bottomSheet.bottomSheetArrow.setImageResource(R.drawable.icn_chevron_up)
                    }
                    else -> {
                        // do nothing.
                    }
                }
            }

            override fun onSlide(bottomSheet: View, slideOffset: Float) {
                // no func
            }

        })
        binding.bottomSheet.threads.text = numThread.toString()
        binding.bottomSheet.minus.setOnClickListener {
            if (numThread <= 1) return@setOnClickListener
            numThread--
            binding.bottomSheet.threads.text = numThread.toString()
            createClassifier()
        }
        binding.bottomSheet.plus.setOnClickListener {
            if (numThread >= 4) return@setOnClickListener
            numThread++
            binding.bottomSheet.threads.text = numThread.toString()
            createClassifier()
        }
        binding.bottomSheet.btnClearModelState.setOnClickListener {
            videoClassifier?.reset()
        }
        initSpinner()

        // Start the camera.
        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(
                this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS
            )
        }
    }

    /**
     * Initialize the spinner to let users change the TFLite model.
     */
    private fun initSpinner() {
        ArrayAdapter.createFromResource(
            this,
            R.array.tfe_pe_models_array,
            android.R.layout.simple_spinner_item
        ).also { adapter ->
            // Specify the layout to use when the list of choices appears
            adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
            // Apply the adapter to the spinner
            binding.bottomSheet.spnSelectModel.adapter = adapter
            binding.bottomSheet.spnSelectModel.setSelection(modelPosition)
        }
        binding.bottomSheet.spnSelectModel.onItemSelectedListener = changeModelListener
    }

    /**
     * Start the image capturing pipeline.
     */
    private fun startCamera() {
        executor = Executors.newSingleThreadExecutor()
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            // Used to bind the lifecycle of cameras to the lifecycle owner
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            // Select back camera as a default
            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            // Create a Preview to show the image captured by the camera on screen.
            val preview = Preview.Builder()
                .build()
                .also {
                    it.setSurfaceProvider(binding.preview.surfaceProvider)
                }

            try {
                // Unbind use cases before rebinding.
                cameraProvider.unbindAll()

                // Create an ImageAnalysis to continuously capture still images using the camera,
                // and feed them to the TFLite model. We set the capturing frame rate to a multiply
                // of the TFLite model's desired FPS to keep the preview smooth, then drop
                // unnecessary frames during image analysis.
                val targetFpsMultiplier = MAX_CAPTURE_FPS.div(MODEL_FPS)
                val targetCaptureFps = MODEL_FPS * targetFpsMultiplier
                val builder = ImageAnalysis.Builder()
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                val extender: Camera2Interop.Extender<*> = Camera2Interop.Extender(builder)
                extender.setCaptureRequestOption(
                    CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE,
                    Range(targetCaptureFps, targetCaptureFps)
                )
                val imageAnalysis = builder.build()

                imageAnalysis.setAnalyzer(executor) { imageProxy ->
                    processImage(imageProxy)
                }

                // Combine the ImageAnalysis and Preview into a use case group.
                val useCaseGroup = UseCaseGroup.Builder()
                    .addUseCase(preview)
                    .addUseCase(imageAnalysis)
                    .setViewPort(binding.preview.viewPort!!)
                    .build()

                // Bind use cases to camera.
                cameraProvider.bindToLifecycle(
                    this, cameraSelector, useCaseGroup
                )

            } catch (e: Exception) {
                Log.e(TAG, "Use case binding failed.", e)
            }

        }, ContextCompat.getMainExecutor(this))
    }

    /**
     * Run a frames received from the camera through the TFLite video classification pipeline.
     */
    private fun processImage(imageProxy: ImageProxy) {
        // Ensure that only one frame is processed at any given moment.
        synchronized(lock) {
            val currentTime = SystemClock.uptimeMillis()
            val diff = currentTime - lastInferenceStartTime

            // Check to ensure that we only run inference at a frequency required by the
            // model, within an acceptable error range (e.g. 10%). Discard the frames
            // that comes too early.
            if (diff * MODEL_FPS >= 1000 /* milliseconds */ * (1 - MODEL_FPS_ERROR_RANGE)) {
                lastInferenceStartTime = currentTime

                val image = imageProxy.image
                image?.let {
                    videoClassifier?.let { classifier ->
                        // Convert the captured frame to Bitmap.
                        val imageBitmap = Bitmap.createBitmap(
                            it.width,
                            it.height,
                            Bitmap.Config.ARGB_8888
                        )
                        CalculateUtils.yuvToRgb(image, imageBitmap)

                        // Rotate the image to the correct orientation.
                        val rotateMatrix = Matrix()
                        rotateMatrix.postRotate(
                            imageProxy.imageInfo.rotationDegrees.toFloat()
                        )
                        val rotatedBitmap = Bitmap.createBitmap(
                            imageBitmap, 0, 0, it.width, it.height,
                            rotateMatrix, false
                        )

                        // Run inference using the TFLite model.
                        val startTimeForReference = SystemClock.uptimeMillis()
                        val results = classifier.classify(rotatedBitmap)
                        val endTimeForReference =
                            SystemClock.uptimeMillis() - startTimeForReference
                        val inputFps = 1000f / diff
                        showResults(results, endTimeForReference, inputFps)

                        if (inputFps < MODEL_FPS * (1 - MODEL_FPS_ERROR_RANGE)) {
                            Log.w(
                                TAG, "Current input FPS ($inputFps) is " +
                                        "significantly lower than the TFLite model's " +
                                        "expected FPS ($MODEL_FPS). It's likely because " +
                                        "model inference takes too long on this device."
                            )
                        }
                    }
                }
            }
            imageProxy.close()
        }
    }

    /**
     * Check whether camera permission is already granted.
     */
    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(
            baseContext, it
        ) == PackageManager.PERMISSION_GRANTED
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                startCamera()
            } else {
                Toast.makeText(this, "Permissions not granted by the user.", Toast.LENGTH_LONG)
                    .show()
            }
        }
    }

    /**
     * Initialize the TFLite video classifier.
     */
    private fun createClassifier() {
        synchronized(lock) {
            if (videoClassifier != null) {
                videoClassifier?.close()
                videoClassifier = null
            }
            val options =
                VideoClassifier.VideoClassifierOptions.builder()
                    .setMaxResult(MAX_RESULT)
                    .setNumThreads(numThread)
                    .build()
            val modelFile = when (modelPosition) {
                0 -> MODEL_MOVINET_A0_FILE
                1 -> MODEL_MOVINET_A1_FILE
                else -> MODEL_MOVINET_A2_FILE
            }

            videoClassifier = VideoClassifier.createFromFileAndLabelsAndOptions(
                this,
                modelFile,
                MODEL_LABEL_FILE,
                options
            )

            // show input size of video classification
            videoClassifier?.getInputSize()?.let {
                binding.bottomSheet.inputSizeInfo.text =
                    getString(R.string.frame_size, it.width, it.height)
            }
            Log.d(TAG, "Classifier created.")
        }
    }

    /**
     * Show the video classification results on the screen.
     */
    private fun showResults(labels: List<Category>, inferenceTime: Long, inputFps: Float) {
        runOnUiThread {
            binding.bottomSheet.tvDetectedItem0.text = labels[0].label
            binding.bottomSheet.tvDetectedItem1.text = labels[1].label
            binding.bottomSheet.tvDetectedItem2.text = labels[2].label
            binding.bottomSheet.tvDetectedItem0Value.text = labels[0].score.toString()
            binding.bottomSheet.tvDetectedItem1Value.text = labels[1].score.toString()
            binding.bottomSheet.tvDetectedItem2Value.text = labels[2].score.toString()
            binding.bottomSheet.inferenceInfo.text =
                getString(R.string.inference_time, inferenceTime)
            binding.bottomSheet.inputFpsInfo.text = String.format("%.1f", inputFps)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        videoClassifier?.close()
        executor.shutdown()
    }
}
