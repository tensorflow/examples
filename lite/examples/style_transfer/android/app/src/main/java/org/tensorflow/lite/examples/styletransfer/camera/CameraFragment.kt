/*
 * Copyright 2019 Google LLC
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

package org.tensorflow.lite.examples.styletransfer.camera

import android.annotation.SuppressLint
import android.content.Context
import android.hardware.display.DisplayManager
import android.net.Uri
import android.os.Bundle
import android.util.DisplayMetrics
import android.util.Log
import android.util.Size
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.FrameLayout
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import androidx.lifecycle.Observer
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.launch
import org.tensorflow.lite.examples.styletransfer.R
import java.io.File
import java.lang.Runnable
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

class CameraFragment : Fragment() {

  // interface to interact with the hosting activity
  interface OnCaptureFinished {
    fun onCaptureFinished(file: File)
  }

  private lateinit var container: FrameLayout
  private lateinit var viewFinder: PreviewView
  private lateinit var callback: OnCaptureFinished

  /** Live data listener for changes in the device orientation relative to the camera */
  private lateinit var relativeOrientation: OrientationLiveData

  private var displayId: Int = -1
  private var lensFacing: Int = CameraSelector.LENS_FACING_FRONT
  private var preview: Preview? = null
  private var imageCapture: ImageCapture? = null
  private var camera: Camera? = null

  private val displayManager by lazy {
    requireContext().getSystemService(Context.DISPLAY_SERVICE) as DisplayManager
  }

  /** Blocking camera operations are performed using this executor */
  private lateinit var cameraExecutor: ExecutorService
  private lateinit var fragmentScope: CoroutineScope
  private val fragmentJob = Job()

  /**
   * We need a display listener for orientation changes that do not trigger a configuration
   * change, for example if we choose to override config change in manifest or for 180-degree
   * orientation changes.
   */
  private val displayListener = object : DisplayManager.DisplayListener {
    override fun onDisplayAdded(displayId: Int) = Unit
    override fun onDisplayRemoved(displayId: Int) = Unit
    override fun onDisplayChanged(displayId: Int) = view?.let { view ->
      if (displayId == this@CameraFragment.displayId) {
        Log.d(TAG, "Rotation changed: ${view.display.rotation}")
        imageCapture?.targetRotation = view.display.rotation
      }
    } ?: Unit
  }

  override fun onDestroyView() {
    super.onDestroyView()

    // Shut down our background executor
    fragmentJob.cancel()

    // Unregister the listeners
    displayManager.unregisterDisplayListener(displayListener)
  }

  /**
  Keeping a reference to the activity to make communication between it and this fragment
  easier.
   */
  override fun onAttach(context: Context) {
    super.onAttach(context)
    callback = context as OnCaptureFinished
  }

  override fun onCreateView(
    inflater: LayoutInflater,
    container: ViewGroup?,
    savedInstanceState: Bundle?
  ): View? =
    inflater.inflate(R.layout.fragment_camera, container, false)

  @SuppressLint("MissingPermission")
  override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
    super.onViewCreated(view, savedInstanceState)
    container = view as FrameLayout
    viewFinder = container.findViewById(R.id.camera_preview)

    // Initialize our background executor
    cameraExecutor = Executors.newSingleThreadExecutor()
    fragmentScope = CoroutineScope(cameraExecutor.asCoroutineDispatcher() + fragmentJob)

    // Every time the orientation of device changes, update rotation for use cases
    displayManager.registerDisplayListener(displayListener, null)

    relativeOrientation = OrientationLiveData(requireContext()).apply {
      observe(
        viewLifecycleOwner,
        Observer { orientation ->
          imageCapture?.targetRotation = orientation
        }
      )
    }

    // Wait for the views to be properly laid out
    viewFinder.post {

      // Keep track of the display in which this view is attached
      displayId = viewFinder.display.displayId

      // Bind use cases
      bindCameraUseCases()
    }
  }

  fun setFacingCamera(lensFacing: Int) {
    this.lensFacing = lensFacing
  }

  fun takePicture() {
    // Get a stable reference of the modifiable image capture use case
    imageCapture?.let { imageCapture ->

      // Create output file to hold the image
      val photoFile = createFile(requireContext())

      // Setup image capture metadata
      val metadata = ImageCapture.Metadata().apply {

        // Mirror image when using the front camera
        isReversedHorizontal = lensFacing == CameraSelector.LENS_FACING_FRONT
      }

      // Create output options object which contains file + metadata
      val outputOptions = ImageCapture.OutputFileOptions.Builder(photoFile)
        .setMetadata(metadata)
        .build()

      // Setup image capture listener which is triggered after photo has been taken
      imageCapture.takePicture(
        outputOptions, cameraExecutor, object : ImageCapture.OnImageSavedCallback {
          override fun onError(exc: ImageCaptureException) {
            Log.e(TAG, "Photo capture failed: ${exc.message}", exc)
          }

          override fun onImageSaved(output: ImageCapture.OutputFileResults) {
            val savedUri = output.savedUri ?: Uri.fromFile(photoFile)
            Log.d(TAG, "Photo capture succeeded: $savedUri")

            fragmentScope.launch(Dispatchers.Main) {
              callback.onCaptureFinished(photoFile)
            }
          }
        })
    }
  }


  /** Declare and bind preview and capture use cases */
  private fun bindCameraUseCases() {

    // Get screen metrics used to setup camera for full screen resolution
    val metrics = DisplayMetrics().also { viewFinder.display.getRealMetrics(it) }
    Log.d(TAG, "Screen metrics: ${metrics.widthPixels} x ${metrics.heightPixels}")

    val screenAspectRatio = aspectRatio(metrics.widthPixels, metrics.heightPixels)
    Log.d(TAG, "Preview aspect ratio: $screenAspectRatio")

    val rotation = viewFinder.display.rotation

    // Bind the CameraProvider to the LifeCycleOwner
    val cameraSelector = CameraSelector.Builder().requireLensFacing(lensFacing).build()
    val cameraProviderFuture = ProcessCameraProvider.getInstance(requireContext())
    cameraProviderFuture.addListener(Runnable {

      // CameraProvider
      val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

      // Preview
      preview = Preview.Builder()
        // We request aspect ratio but no resolution
        .setTargetAspectRatio(screenAspectRatio)
        // Set initial target rotation
        .setTargetRotation(rotation)
        .build()

      // ImageCapture
      imageCapture = ImageCapture.Builder()
        .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
        .setTargetResolution(Size(512, 512))
        // Set initial target rotation, we will have to call this again if rotation changes
        // during the lifecycle of this use case
        .setTargetRotation(rotation)
        .build()

      // Must unbind the use-cases before rebinding them
      cameraProvider.unbindAll()

      try {
        // A variable number of use-cases can be passed here -
        // camera provides access to CameraControl & CameraInfo
        camera = cameraProvider.bindToLifecycle(
          this, cameraSelector, preview, imageCapture
        )

        // Attach the viewfinder's surface provider to preview use case
        preview?.setSurfaceProvider(viewFinder.createSurfaceProvider(camera?.cameraInfo))
      } catch (exc: Exception) {
        Log.e(TAG, "Use case binding failed", exc)
      }

    }, ContextCompat.getMainExecutor(requireContext()))
  }

  /**
   *  Detecting the most suitable ratio for dimensions provided in @params by counting absolute
   *  of preview ratio to one of the provided values.
   *
   *  @param width - preview width
   *  @param height - preview height
   *  @return suitable aspect ratio
   */
  private fun aspectRatio(width: Int, height: Int): Int {
    val previewRatio = max(width, height).toDouble() / min(width, height)
    if (abs(previewRatio - RATIO_4_3_VALUE) <= abs(previewRatio - RATIO_16_9_VALUE)) {
      return AspectRatio.RATIO_4_3
    }
    return AspectRatio.RATIO_16_9
  }

  companion object {

    private const val TAG = "CameraXBasic"
    private const val FILENAME = "yyyy-MM-dd-HH-mm-ss-SSS"
    private const val PHOTO_EXTENSION = "jpg"
    private const val RATIO_4_3_VALUE = 4.0 / 3.0
    private const val RATIO_16_9_VALUE = 16.0 / 9.0

    /**
     * Create a [File] named a using formatted timestamp with the current date and time.
     *
     * @return [File] created.
     */
    private fun createFile(context: Context): File {
      val sdf = SimpleDateFormat(FILENAME, Locale.US)
      return File(context.filesDir, "IMG_${sdf.format(Date())}.$PHOTO_EXTENSION")
    }

    @JvmStatic
    fun newInstance(): CameraFragment =
      CameraFragment()
  }
}