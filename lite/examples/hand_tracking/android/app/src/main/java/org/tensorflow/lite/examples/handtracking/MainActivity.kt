package org.tensorflow.lite.examples.handtracking

import android.app.Activity
import android.content.Intent
import android.graphics.*
import android.net.Uri
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import android.widget.Button
import android.widget.ImageView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.FileProvider
import androidx.exifinterface.media.ExifInterface
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import org.tensorflow.lite.examples.handtracking.handlandmark.HandDetector
import org.tensorflow.lite.examples.handtracking.handlandmark.data.HandLandmark
import java.io.File
import java.io.IOException
import java.text.SimpleDateFormat
import java.util.*
import kotlin.math.max
import kotlin.math.min

class MainActivity : AppCompatActivity() {
    companion object {
        private const val TAG = "Hand detection"
        private const val REQUEST_IMAGE_CAPTURE = 1

        // This list defines the lines that are drawn when visualizing the hand landmark detection
        // results. These lines connect:
        // landmarkConnections[2*n] and landmarkConnections[2*n+1]
        private val landmarkConnections = listOf(
            HandLandmark.WRIST,
            HandLandmark.THUMB_CMC,
            HandLandmark.THUMB_CMC,
            HandLandmark.THUMB_MCP,
            HandLandmark.THUMB_MCP,
            HandLandmark.THUMB_IP,
            HandLandmark.THUMB_IP,
            HandLandmark.THUMB_TIP,
            HandLandmark.WRIST,
            HandLandmark.INDEX_FINGER_MCP,
            HandLandmark.INDEX_FINGER_MCP,
            HandLandmark.INDEX_FINGER_PIP,
            HandLandmark.INDEX_FINGER_PIP,
            HandLandmark.INDEX_FINGER_DIP,
            HandLandmark.INDEX_FINGER_DIP,
            HandLandmark.INDEX_FINGER_TIP,
            HandLandmark.INDEX_FINGER_MCP,
            HandLandmark.MIDDLE_FINGER_MCP,
            HandLandmark.MIDDLE_FINGER_MCP,
            HandLandmark.MIDDLE_FINGER_PIP,
            HandLandmark.MIDDLE_FINGER_PIP,
            HandLandmark.MIDDLE_FINGER_DIP,
            HandLandmark.MIDDLE_FINGER_DIP,
            HandLandmark.MIDDLE_FINGER_TIP,
            HandLandmark.MIDDLE_FINGER_MCP,
            HandLandmark.RING_FINGER_MCP,
            HandLandmark.RING_FINGER_MCP,
            HandLandmark.RING_FINGER_PIP,
            HandLandmark.RING_FINGER_PIP,
            HandLandmark.RING_FINGER_DIP,
            HandLandmark.RING_FINGER_DIP,
            HandLandmark.RING_FINGER_TIP,
            HandLandmark.RING_FINGER_MCP,
            HandLandmark.PINKY_MCP,
            HandLandmark.WRIST,
            HandLandmark.PINKY_MCP,
            HandLandmark.PINKY_MCP,
            HandLandmark.PINKY_PIP,
            HandLandmark.PINKY_PIP,
            HandLandmark.PINKY_DIP,
            HandLandmark.PINKY_DIP,
            HandLandmark.PINKY_TIP
        )
    }

    private lateinit var captureImageFab: Button
    private lateinit var inputImageView: ImageView
    private lateinit var currentPhotoPath: String
    private lateinit var handDetector: HandDetector

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        handDetector = HandDetector.create(this)
        captureImageFab = findViewById(R.id.captureImageFab)
        inputImageView = findViewById(R.id.imageView)

        captureImageFab.setOnClickListener {
            dispatchTakePictureIntent()
        }

    }

    /**
     * Run hand landmark detection on the input image
     */
    private fun handLandmarkDetection(bitmap: Bitmap) {
        val landmarks = handDetector.process(bitmap)
        if (landmarks.isNotEmpty()) {
            showLandmarks(bitmap, landmarks).let {
                runOnUiThread {
                    inputImageView.setImageBitmap(it)
                }
            }
        }
    }

    /**
     * Make a copy of the input image and draw hand landmarks on top ut.
     */
    private fun showLandmarks(inputImage: Bitmap, landmarks: List<HandLandmark>): Bitmap {
        val drawBitmap = inputImage.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(drawBitmap)
        val penStroke = Paint().apply {
            color = Color.GREEN
            strokeWidth = 20f
        }
        val penPoint = Paint().apply {
            color = Color.RED
            strokeWidth = 20f
            style = Paint.Style.FILL
        }
        val lines = mutableListOf<Float>()
        val points = mutableListOf<Float>()
        for (i in landmarkConnections.indices step 2) {
            val startX =
                landmarks[landmarkConnections[i]].x * drawBitmap.width
            val startY =
                landmarks[landmarkConnections[i]].y * drawBitmap.height
            val endX =
                landmarks[landmarkConnections[i + 1]].x * drawBitmap.width
            val endY =
                landmarks[landmarkConnections[i + 1]].y * drawBitmap.height

            lines.add(startX)
            lines.add(startY)
            lines.add(endX)
            lines.add(endY)
            points.add(startX)
            points.add(startY)
        }
        canvas.drawLines(lines.toFloatArray(), penStroke)
        canvas.drawPoints(points.toFloatArray(), penPoint)
        return drawBitmap
    }

    private fun dispatchTakePictureIntent() {
        Intent(MediaStore.ACTION_IMAGE_CAPTURE).also { takePictureIntent ->
            // Ensure that there's a camera activity to handle the intent
            takePictureIntent.resolveActivity(packageManager)?.also {
                // Create the File where the photo should go
                val photoFile: File? = try {
                    createImageFile()
                } catch (e: IOException) {
                    Log.e(TAG, e.message.toString())
                    null
                }
                // Continue only if the File was successfully created
                photoFile?.also {
                    val photoURI: Uri = FileProvider.getUriForFile(
                        this,
                        "org.tensorflow.lite.examples.handtracking.fileprovider",
                        it
                    )
                    takePictureIntent.putExtra(MediaStore.EXTRA_OUTPUT, photoURI)
                    startActivityForResult(takePictureIntent, REQUEST_IMAGE_CAPTURE)
                }
            }
        }
    }

    @Throws(IOException::class)
    private fun createImageFile(): File {
        // Create an image file name
        val timeStamp: String = SimpleDateFormat("yyyyMMdd_HHmmss").format(Date())
        val storageDir: File? = getExternalFilesDir(Environment.DIRECTORY_PICTURES)
        return File.createTempFile(
            "JPEG_${timeStamp}_", /* prefix */
            ".jpg", /* suffix */
            storageDir /* directory */
        ).apply {
            // Save a file: path for use with ACTION_VIEW intents
            currentPhotoPath = absolutePath
        }
    }

    /**
     * getCapturedImage():
     *     Decodes and crops the captured image from camera.
     */
    private fun getCapturedImage(): Bitmap {
        // Get the dimensions of the View
        val targetW: Int = inputImageView.width
        val targetH: Int = inputImageView.height

        val bmOptions = BitmapFactory.Options().apply {
            // Get the dimensions of the bitmap
            inJustDecodeBounds = true

            BitmapFactory.decodeFile(currentPhotoPath, this)

            val photoW: Int = outWidth
            val photoH: Int = outHeight

            // Determine how much to scale down the image
            val scaleFactor: Int = max(1, min(photoW / targetW, photoH / targetH))

            // Decode the image file into a Bitmap sized to fill the View
            inJustDecodeBounds = false
            inSampleSize = scaleFactor
            inMutable = true
        }
        val exifInterface = ExifInterface(currentPhotoPath)
        val orientation = exifInterface.getAttributeInt(
            ExifInterface.TAG_ORIENTATION,
            ExifInterface.ORIENTATION_UNDEFINED
        )

        val bitmap = BitmapFactory.decodeFile(currentPhotoPath, bmOptions)
        return when (orientation) {
            ExifInterface.ORIENTATION_ROTATE_90 -> {
                rotateImage(bitmap, 90f)
            }
            ExifInterface.ORIENTATION_ROTATE_180 -> {
                rotateImage(bitmap, 180f)
            }
            ExifInterface.ORIENTATION_ROTATE_270 -> {
                rotateImage(bitmap, 270f)
            }
            else -> {
                bitmap
            }
        }
    }

    private fun rotateImage(source: Bitmap, angle: Float): Bitmap {
        val matrix = Matrix()
        matrix.postRotate(angle)
        return Bitmap.createBitmap(
            source, 0, 0, source.width, source.height,
            matrix, true
        )
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == REQUEST_IMAGE_CAPTURE &&
            resultCode == Activity.RESULT_OK
        ) {
            lifecycleScope.launch(Dispatchers.Default) {
                getCapturedImage().let {
                    runOnUiThread {
                        inputImageView.setImageBitmap(it)
                    }
                    handLandmarkDetection(it)
                }
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        handDetector.close()
    }
}
