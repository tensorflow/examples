/*
 * Copyright 2022 The TensorFlow Authors
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

package org.tensorflow.lite.examples.classification.playservices

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
import android.util.Log
import android.view.View
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.AspectRatio
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.google.android.gms.tasks.Task
import com.google.android.gms.tflite.java.TfLite
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlin.random.Random
import org.tensorflow.lite.examples.classification.playservices.databinding.ActivityCameraBinding

/** Activity that displays the camera and performs object detection on the incoming frames */
class CameraActivity : AppCompatActivity() {

  private lateinit var activityCameraBinding: ActivityCameraBinding

  private lateinit var bitmapBuffer: Bitmap

  private val executor = Executors.newSingleThreadExecutor()
  private val permissions = listOf(Manifest.permission.CAMERA)
  private val permissionsRequestCode = Random.nextInt(0, 10000)

  private var lensFacing: Int = CameraSelector.LENS_FACING_BACK
  private val isFrontFacing
    get() = lensFacing == CameraSelector.LENS_FACING_FRONT

  private var pauseAnalysis = false
  private var imageRotationDegrees: Int = 0

  // Initialize TFLite once. Must be called before creating the classifier
  private val initializeTask: Task<Void> by lazy { TfLite.initialize(this) }
  private var classifier: ImageClassificationHelper? = null

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    activityCameraBinding = ActivityCameraBinding.inflate(layoutInflater)
    setContentView(activityCameraBinding.root)

    // Initialize TFLite asynchronously
    initializeTask
      .addOnSuccessListener {
        Log.d(TAG, "TFLite in Play Services initialized successfully.")
        classifier = ImageClassificationHelper(this)
      }
      .addOnFailureListener { e -> Log.e(TAG, "TFLite in Play Services failed to initialize.", e) }

    activityCameraBinding.cameraCaptureButton.setOnClickListener {
      // Disable all camera controls
      it.isEnabled = false
      if (pauseAnalysis) {
        // If image analysis is in paused state, resume it
        pauseAnalysis = false
        activityCameraBinding.imagePredicted.visibility = View.GONE
      } else {
        // Otherwise, pause image analysis and freeze image
        pauseAnalysis = true
        val matrix =
          Matrix().apply {
            postRotate(imageRotationDegrees.toFloat())
            if (isFrontFacing) postScale(-1f, 1f)
          }
        val uprightImage =
          Bitmap.createBitmap(
            bitmapBuffer,
            0,
            0,
            bitmapBuffer.width,
            bitmapBuffer.height,
            matrix,
            true
          )
        activityCameraBinding.imagePredicted.setImageBitmap(uprightImage)
        activityCameraBinding.imagePredicted.visibility = View.VISIBLE
      }

      // Re-enable camera controls
      it.isEnabled = true
    }
  }

  override fun onDestroy() {
    // Terminate all outstanding analyzing jobs (if there is any).
    executor.apply {
      shutdown()
      awaitTermination(1000, TimeUnit.MILLISECONDS)
    }
    // Release TFLite resources
    classifier?.close()
    super.onDestroy()
  }

  /** Declare and bind preview and analysis use cases */
  @SuppressLint("UnsafeExperimentalUsageError")
  private fun bindCameraUseCases() =
    activityCameraBinding.viewFinder.post {
      val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
      cameraProviderFuture.addListener(
        {
          // Camera provider is now guaranteed to be available
          val cameraProvider = cameraProviderFuture.get()

          // Set up the view finder use case to display camera preview
          val preview =
            Preview.Builder()
              .setTargetAspectRatio(AspectRatio.RATIO_4_3)
              .setTargetRotation(activityCameraBinding.viewFinder.display.rotation)
              .build()

          // Set up the image analysis use case which will process frames in real time
          val imageAnalysis =
            ImageAnalysis.Builder()
              .setTargetAspectRatio(AspectRatio.RATIO_4_3)
              .setTargetRotation(activityCameraBinding.viewFinder.display.rotation)
              .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
              .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
              .build()

          var frameCounter = 0
          var lastFpsTimestamp = System.currentTimeMillis()

          imageAnalysis.setAnalyzer(
            executor,
            ImageAnalysis.Analyzer { image ->
              if (!::bitmapBuffer.isInitialized) {
                // The image rotation and RGB image buffer are initialized only once
                // the analyzer has started running
                imageRotationDegrees = image.imageInfo.rotationDegrees
                bitmapBuffer =
                  Bitmap.createBitmap(image.width, image.height, Bitmap.Config.ARGB_8888)
              }

              // Early exit: image analysis is in paused state, or TFLite is not initialized
              if (pauseAnalysis || classifier == null) {
                image.close()
                return@Analyzer
              }

              // Copy out RGB bits to our shared buffer
              image.use { bitmapBuffer.copyPixelsFromBuffer(image.planes[0].buffer) }

              // Perform the image classification for the current frame
              val recognitions = classifier?.classify(bitmapBuffer, imageRotationDegrees)

              reportRecognition(recognitions)

              // Compute the FPS of the entire pipeline
              val frameCount = 10
              if (++frameCounter % frameCount == 0) {
                frameCounter = 0
                val now = System.currentTimeMillis()
                val delta = now - lastFpsTimestamp
                val fps = 1000 * frameCount.toFloat() / delta
                Log.d(TAG, "FPS: ${"%.02f".format(fps)}")
                lastFpsTimestamp = now
              }
            }
          )

          // Create a new camera selector each time, enforcing lens facing
          val cameraSelector = CameraSelector.Builder().requireLensFacing(lensFacing).build()

          // Apply declared configs to CameraX using the same lifecycle owner
          cameraProvider.unbindAll()
          cameraProvider.bindToLifecycle(
            this as LifecycleOwner,
            cameraSelector,
            preview,
            imageAnalysis
          )

          // Use the camera object to link our preview use case with the view
          preview.setSurfaceProvider(activityCameraBinding.viewFinder.surfaceProvider)
        },
        ContextCompat.getMainExecutor(this)
      )
    }

  /** Displays recognition results on screen. */
  private fun reportRecognition(
    recognitions: List<ImageClassificationHelper.Recognition>?,
  ) =
    activityCameraBinding.viewFinder.post {

      // Early exit: if recognition is null, or there are not enough recognition results.
      if (recognitions == null || recognitions.size < MAX_REPORT) {
        activityCameraBinding.textPrediction.visibility = View.GONE
        return@post
      }

      // Update the text and UI
      activityCameraBinding.textPrediction.text =
        recognitions.subList(0, MAX_REPORT).joinToString(separator = "\n") {
          "${"%.2f".format(it.confidence)} ${it.title}"
        }

      // Make sure all UI elements are visible
      activityCameraBinding.textPrediction.visibility = View.VISIBLE
    }

  override fun onResume() {
    super.onResume()

    // Request permissions each time the app resumes, since they can be revoked at any time
    if (!hasPermissions(this)) {
      ActivityCompat.requestPermissions(this, permissions.toTypedArray(), permissionsRequestCode)
    } else {
      bindCameraUseCases()
    }
  }

  override fun onRequestPermissionsResult(
    requestCode: Int,
    permissions: Array<out String>,
    grantResults: IntArray,
  ) {
    super.onRequestPermissionsResult(requestCode, permissions, grantResults)
    if (requestCode == permissionsRequestCode && hasPermissions(this)) {
      bindCameraUseCases()
    } else {
      finish() // If we don't have the required permissions, we can't run
    }
  }

  /** Convenience method used to check if all permissions required by this app are granted */
  private fun hasPermissions(context: Context) =
    permissions.all {
      ContextCompat.checkSelfPermission(context, it) == PackageManager.PERMISSION_GRANTED
    }

  companion object {
    private val TAG = CameraActivity::class.java.simpleName
    private const val MAX_REPORT = 3
  }
}
