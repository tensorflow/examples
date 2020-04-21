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
import androidx.lifecycle.Observer
import android.content.Context
import android.graphics.ImageFormat
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CameraMetadata
import android.hardware.camera2.CaptureRequest
import android.hardware.camera2.CaptureResult
import android.hardware.camera2.TotalCaptureResult
import android.media.Image
import android.media.ImageReader
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import androidx.fragment.app.Fragment
import android.util.Log
import android.util.Size
import android.view.LayoutInflater
import android.view.Surface
import android.view.SurfaceHolder
import android.view.SurfaceView
import android.view.View
import android.view.ViewGroup
import java.io.Closeable
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.ArrayBlockingQueue
import java.util.concurrent.TimeoutException
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException
import kotlin.coroutines.suspendCoroutine
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import org.tensorflow.lite.examples.styletransfer.ImageUtils

@SuppressWarnings("GoodTime")
class CameraFragment : Fragment() {

  private var cameraInitialized: Boolean = false
  private val fragmentJob = SupervisorJob()
  val fragmentScope = CoroutineScope(Dispatchers.Main + fragmentJob)

  // interface to interact with the hosting activity
  interface OnCaptureFinished {
    fun onCaptureFinished(file: File)
  }

  private var cameraFacing = CameraCharacteristics.LENS_FACING_FRONT
  private var cameraId = "0"

  /**
   * define the aspect ratio that will be used by the camera
   */
  private val aspectRatio = Size(4, 3)

  /**
   * Max preview width that is guaranteed by Camera2 API
   */
  private val MAX_PREVIEW_WIDTH = 1080

  internal lateinit var callback: OnCaptureFinished

  /** Detects, characterizes, and connects to a CameraDevice (used for all camera operations) */
  private val cameraManager: CameraManager by lazy {
    val context = requireContext().applicationContext
    context.getSystemService(Context.CAMERA_SERVICE) as CameraManager
  }

  /** [CameraCharacteristics] corresponding to the provided Camera ID */
  private lateinit var characteristics: CameraCharacteristics

  /** Readers used as buffers for camera still shots */
  private lateinit var imageReader: ImageReader

  /** [HandlerThread] where all camera operations run */
  private val cameraThread = HandlerThread("CameraThread").apply { start() }

  /** [Handler] corresponding to [cameraThread] */
  private val cameraHandler = Handler(cameraThread.looper)

  /** [HandlerThread] where all buffer reading operations run */
  private val imageReaderThread = HandlerThread("imageReaderThread").apply { start() }

  /** [Handler] corresponding to [imageReaderThread] */
  private val imageReaderHandler = Handler(imageReaderThread.looper)

  /** Where the camera preview is displayed */
  private lateinit var viewFinder: SurfaceView

  /** Internal reference to the ongoing [CameraCaptureSession] configured with our parameters */
  private lateinit var session: CameraCaptureSession

  /** Live data listener for changes in the device orientation relative to the camera */
  private lateinit var relativeOrientation: OrientationLiveData

  override fun onCreateView(
    inflater: LayoutInflater,
    container: ViewGroup?,
    savedInstanceState: Bundle?
  ): View? = AutoFitSurfaceView(requireContext())

  @SuppressLint("MissingPermission")
  override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
    super.onViewCreated(view, savedInstanceState)
    viewFinder = view as AutoFitSurfaceView

    cameraId = getCameraId()
    characteristics = cameraManager.getCameraCharacteristics(cameraId)
    val size = characteristics.get(
      CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP
    )!!
      .getOutputSizes(ImageFormat.JPEG)
      .filter {
        it.width <= MAX_PREVIEW_WIDTH && verifyAspectRatio(it.width, it.height, aspectRatio)
      }.first()
    imageReader = ImageReader.newInstance(
      size.width, size.height,
      ImageFormat.JPEG, IMAGE_BUFFER_SIZE
    )

    view.holder.addCallback(object : SurfaceHolder.Callback {
      override fun surfaceDestroyed(holder: SurfaceHolder) = Unit

      override fun surfaceChanged(
        holder: SurfaceHolder,
        format: Int,
        width: Int,
        height: Int
      ) = Unit

      override fun surfaceCreated(holder: SurfaceHolder) {
        // Selects appropriate preview size and configures view finder
        val previewSize = getPreviewOutputSize(
          viewFinder.display, characteristics, SurfaceHolder::class.java, aspectRatio
        )
        Log.d(TAG, "View finder size: ${viewFinder.width} x ${viewFinder.height}")
        Log.d(TAG, "Selected preview size: $previewSize")
        view.holder.setFixedSize(previewSize.width, previewSize.height)
        view.setAspectRatio(previewSize.width, previewSize.height)

        // To ensure that size is set, initialize camera in the view's thread
        view.post { initializeCamera() }
      }
    })

    // Used to rotate the output media to match device orientation
    relativeOrientation = OrientationLiveData(requireContext(), characteristics).apply {
      observe(
        this@CameraFragment,
        Observer {
          orientation ->
          Log.d(TAG, "Orientation changed: $orientation")
        }
      )
    }
  }

  /**
   * Begin all camera operations in a coroutine in the main thread. This function:
   * - Opens the camera
   * - Configures the camera session
   * - Starts the preview by dispatching a repeating capture request
   * - Sets up the image capture listeners
   */
  private fun initializeCamera() = fragmentScope.launch(Dispatchers.Main) {
    val camera = openCamera()
    session = startCaptureSession(camera)
    cameraInitialized = true
    val captureRequest = camera.createCaptureRequest(
      CameraDevice.TEMPLATE_PREVIEW
    ).apply { addTarget(viewFinder.holder.surface) }

    // This will keep sending the capture request as frequently as possible until the
    // session is torn down or session.stopRepeating() is called
    session.setRepeatingRequest(captureRequest.build(), null, cameraHandler)
  }

  /** Opens the camera and returns the opened device (as the result of the suspend coroutine) */
  @SuppressLint("MissingPermission")
  private suspend fun openCamera(): CameraDevice = suspendCoroutine { cont ->
    val cameraId = getCameraId()
    cameraManager.openCamera(
      cameraId,
      object : CameraDevice.StateCallback() {
        override fun onDisconnected(device: CameraDevice) {
          Log.w(TAG, "Camera $cameraId has been disconnected")
          requireActivity().finish()
        }

        override fun onError(device: CameraDevice, error: Int) {
          val exc = RuntimeException("Camera $cameraId open error: $error")
          Log.e(TAG, exc.message, exc)
          cont.resumeWithException(exc)
        }

        override fun onOpened(device: CameraDevice) = cont.resume(device)
      },
      cameraHandler
    )
  }

  private fun getCameraId(): String {
    // Get list of all compatible cameras
    val cameraIds = cameraManager.cameraIdList.filter {
      val characteristics = cameraManager.getCameraCharacteristics(it)
      val capabilities = characteristics.get(
        CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES
      )
      capabilities?.contains(
        CameraMetadata.REQUEST_AVAILABLE_CAPABILITIES_BACKWARD_COMPATIBLE
      ) ?: false
    }

    // Iterate over the list of cameras and return the one that is facing the selected direction
    cameraIds.forEach { id ->
      val characteristics = cameraManager.getCameraCharacteristics(id)
      if (cameraFacing == characteristics.get(CameraCharacteristics.LENS_FACING)) {
        return id
      }
    }
    return cameraId.first().toString()
  }

  /**
   * Starts a [CameraCaptureSession] and returns the configured session (as the result of the
   * suspend coroutine
   */
  private suspend fun startCaptureSession(device: CameraDevice):
    CameraCaptureSession = suspendCoroutine { cont ->

      // Create list of Surfaces where the camera will output frames
      val targets: MutableList<Surface> =
        arrayOf(viewFinder.holder.surface, imageReader.surface).toMutableList()

      // Create a capture session using the predefined targets; this also involves defining the
      // session state callback to be notified of when the session is ready
      device.createCaptureSession(
        targets,
        object : CameraCaptureSession.StateCallback() {
          override fun onConfigureFailed(session: CameraCaptureSession) {
            val exc = RuntimeException(
              "Camera ${device.id} session configuration failed, see log for details"
            )
            Log.e(TAG, exc.message, exc)
            cont.resumeWithException(exc)
          }

          override fun onConfigured(session: CameraCaptureSession) = cont.resume(session)
        },
        cameraHandler
      )
    }

  fun takePicture() {
    // Perform I/O heavy operations in a different scope
    fragmentScope.launch(Dispatchers.IO) {
      internalTakePicture().use { result ->
        Log.d(TAG, "Result received: $result")

        // Save the result to disk
        val output = saveResult(result)
        Log.d(TAG, "Image saved: ${output.absolutePath}")

        // If the result is a JPEG file, update EXIF metadata with orientation info
        if (output.extension == "jpg") {
          ImageUtils.setExifOrientation(output.absolutePath, result.orientation.toString())
        }

        // Display the photo taken to user
        fragmentScope.launch(Dispatchers.Main) {
          callback.onCaptureFinished(output)
          Log.d(TAG, "almost viewing a picture")
        }
      }
    }
  }

  /**
   * Helper function used to capture a still image using the [CameraDevice.TEMPLATE_STILL_CAPTURE]
   * template. It performs synchronization between the [CaptureResult] and the [Image] resulting
   * from the single capture, and outputs a [CombinedCaptureResult] object.
   */
  private suspend fun internalTakePicture(): CombinedCaptureResult = suspendCoroutine { cont ->

    // Flush any images left in the image reader
    @Suppress("ControlFlowWithEmptyBody")
    while (imageReader.acquireNextImage() != null) {}

    // Start a new image queue
    val imageQueue = ArrayBlockingQueue<Image>(IMAGE_BUFFER_SIZE)
    imageReader.setOnImageAvailableListener(
      { reader ->
        val image = reader.acquireNextImage()
        Log.d(TAG, "Image available in queue: ${image.timestamp}")
        imageQueue.add(image)
      },
      imageReaderHandler
    )

    val captureRequest = session.device.createCaptureRequest(
      CameraDevice.TEMPLATE_STILL_CAPTURE
    ).apply { addTarget(imageReader.surface) }
    session.capture(
      captureRequest.build(),
      object : CameraCaptureSession.CaptureCallback() {
        override fun onCaptureCompleted(
          session: CameraCaptureSession,
          request: CaptureRequest,
          result: TotalCaptureResult
        ) {
          super.onCaptureCompleted(session, request, result)
          val resultTimestamp = result.get(CaptureResult.SENSOR_TIMESTAMP)
          Log.d(TAG, "Capture result received: $resultTimestamp")

          // Set a timeout in case image captured is dropped from the pipeline
          val exc = TimeoutException("Image dequeuing took too long")
          val timeoutRunnable = Runnable { cont.resumeWithException(exc) }
          imageReaderHandler.postDelayed(timeoutRunnable, IMAGE_CAPTURE_TIMEOUT_MILLIS)

          // Loop in the coroutine's context until an image with matching timestamp comes
          // We need to launch the coroutine context again because the callback is done in
          //  the handler provided to the `capture` method, not in our coroutine context
          fragmentScope.launch(cont.context) {
            while (true) {
              // Dequeue images while timestamps don't match
              val image = imageQueue.take()

              Log.d(TAG, "Matching image dequeued: ${image.timestamp}")

              // Unset the image reader listener
              imageReaderHandler.removeCallbacks(timeoutRunnable)
              imageReader.setOnImageAvailableListener(null, null)

              // Clear the queue of images, if there are left
              while (imageQueue.size > 0) {
                imageQueue.take().close()
              }

              // Compute EXIF orientation metadata
              val rotation = relativeOrientation.value ?: 0
              val mirrored = characteristics.get(CameraCharacteristics.LENS_FACING) ==
                CameraCharacteristics.LENS_FACING_FRONT
              val exifOrientation = ImageUtils.computeExifOrientation(rotation, mirrored)

              // Build the result and resume progress
              cont.resume(
                CombinedCaptureResult(
                  image, result, exifOrientation, imageReader.imageFormat
                )
              )

              // There is no need to break out of the loop, this coroutine will suspend
            }
          }
        }
      },
      cameraHandler
    )
  }

  /** Helper function used to save a [CombinedCaptureResult] into a [File] */
  private suspend fun saveResult(result: CombinedCaptureResult): File = suspendCoroutine { cont ->
    // When the format is JPEG or DEPTH JPEG we can simply save the bytes as-is
    when (result.format) {
      ImageFormat.JPEG -> {
        val buffer = result.image.planes[0].buffer
        val bytes = ByteArray(buffer.remaining()).apply { buffer.get(this) }
        try {
          val output = createFile(requireContext(), "jpg")
          FileOutputStream(output).use { it.write(bytes) }
          cont.resume(output)
        } catch (exc: IOException) {
          Log.e(TAG, "Unable to write JPEG image to file", exc)
          cont.resumeWithException(exc)
        }
      }

      // No other formats are supported by this sample
      else -> {
        val exc = RuntimeException("Unknown image format: ${result.image.format}")
        Log.e(TAG, exc.message, exc)
        cont.resumeWithException(exc)
      }
    }
  }

  override fun onStop() {
    super.onStop()
    try {
      session.close()
      session.device.close()
    } catch (exc: Throwable) {
      Log.e(TAG, "Error closing camera", exc)
    }
  }

  override fun onDestroy() {
    super.onDestroy()
    fragmentJob.cancel()
    cameraThread.quitSafely()
    imageReaderThread.quitSafely()
  }

  /**
   Keeping a reference to the activity to make communication between it and this fragment
   easier.
   */
  override fun onAttach(context: Context) {
    super.onAttach(context)
    callback = context as OnCaptureFinished
  }

  fun setFacingCamera(lensFacing: Int) {
    cameraFacing = lensFacing
  }

  companion object {
    private val TAG = CameraFragment::class.java.simpleName

    /** Maximum number of images that will be held in the reader's buffer */
    private const val IMAGE_BUFFER_SIZE: Int = 3

    /** Maximum time allowed to wait for the result of an image capture */
    private const val IMAGE_CAPTURE_TIMEOUT_MILLIS: Long = 5000

    /** Helper data class used to hold capture metadata results with their associated image */
    data class CombinedCaptureResult(
      val image: Image,
      val metadata: CaptureResult,
      val orientation: Int,
      val format: Int
    ) : Closeable {
      override fun close() = image.close()
    }

    /**
     * Create a [File] named a using formatted timestamp with the current date and time.
     *
     * @return [File] created.
     */
    private fun createFile(context: Context, extension: String): File {
      val sdf = SimpleDateFormat("yyyy_MM_dd_HH_mm_ss_SSS", Locale.US)
      return File(context.filesDir, "IMG_${sdf.format(Date())}.$extension")
    }

    @JvmStatic
    fun newInstance(): CameraFragment =
      CameraFragment()
  }
}
