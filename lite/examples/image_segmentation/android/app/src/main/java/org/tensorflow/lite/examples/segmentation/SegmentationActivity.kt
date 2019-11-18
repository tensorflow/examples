package org.tensorflow.lite.examples.segmentation

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.net.Uri
import android.os.*
import android.provider.MediaStore
import android.text.Spannable
import android.text.SpannableStringBuilder
import android.text.style.BackgroundColorSpan
import android.text.style.ForegroundColorSpan
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.widget.SwitchCompat
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.exifinterface.media.ExifInterface
import com.addisonelliott.segmentedbutton.SegmentedButtonGroup
import org.tensorflow.lite.examples.segmentation.env.Logger
import org.tensorflow.lite.examples.segmentation.ext.cropCenter
import org.tensorflow.lite.examples.segmentation.ext.transformUp
import org.tensorflow.lite.examples.segmentation.tflite.ImageSegmentator
import org.tensorflow.lite.examples.segmentation.tflite.SegmentationResult
import java.io.File
import java.io.IOException
import java.io.InputStream
import java.text.SimpleDateFormat
import java.util.*


class SegmentationActivity: AppCompatActivity() {

    private val LOGGER = Logger()
    private var handler: Handler? = null
    private var handlerThread: HandlerThread? = null

    private lateinit var inferenceStatusLabel:TextView
    private lateinit var legendLabel:TextView
    private lateinit var cameraImageView:ImageView
    private lateinit var photoImageView:ImageView
    private lateinit var imageView:ImageView
    private lateinit var cropSwitch: SwitchCompat

    private lateinit var segmentedControl: SegmentedButtonGroup

    /// Image segmentator instance that runs image segmentation.
    private lateinit var imageSegmentator:ImageSegmentator

    /// Target image to run image segmentation on.
    private var targetImage: Bitmap? = null


    /// Processed (e.g center cropped)) image from targetImage that is fed to imageSegmentator.
    private var segmentationInput: Bitmap? = null

    /// Image segmentation result.
    private var segmentationResult: SegmentationResult? = null

    private var pictureImagePath:String? = null

    companion object {
        val SDF = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.ENGLISH)

        const val STORAGE_PERMISSION_REQUEST_CODE = 101
        const val CAMERA_PERMISSION_REQUEST_CODE = 102

        const val CHOOSE_PHOTO_ACTIVITY_REQUEST_CODE = 501
        const val TAKE_PHOTO_ACTIVITY_REQUEST_CODE = 502

        const val ONE_SECOND_IN_MILLIS = 1000L
    }


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_segmentation)

        cameraImageView = findViewById(R.id.iv_camera)
        photoImageView = findViewById(R.id.iv_photo)

        imageView = findViewById(R.id.image)

        segmentedControl = findViewById(R.id.segmented_group)
        segmentedControl.setOnPositionChangedListener {
            onSegmentChanged()
        }

        legendLabel = findViewById(R.id.tv_legend)
        inferenceStatusLabel = findViewById(R.id.tv_inference_time)
        cropSwitch = findViewById(R.id.switch_crop_to_square)

        cropSwitch.setOnCheckedChangeListener{ _, _ ->
            runSegmentation(targetImage!!)
        }

        imageSegmentator = ImageSegmentator(this)

        cameraImageView.setOnClickListener {
            openCamera()
        }

        photoImageView.setOnClickListener {
            openGallery()
        }

        val takePictureIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
        if (takePictureIntent.resolveActivity(packageManager) != null) {
            cameraImageView.alpha = 1.0f
            cameraImageView.isEnabled = true
        } else {
            cameraImageView.alpha = 0.5f
            cameraImageView.isEnabled = false
        }

        Handler().postDelayed({
            runSegmentation(BitmapFactory.decodeResource(resources, R.drawable.boy))
        }, ONE_SECOND_IN_MILLIS)
    }

    private fun openGallery() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.READ_EXTERNAL_STORAGE), STORAGE_PERMISSION_REQUEST_CODE)
            return
        }

        val intent = Intent()
        intent.type = "image/*"
        intent.action = Intent.ACTION_GET_CONTENT
        startActivityForResult(Intent.createChooser(intent, "Select Picture"), CHOOSE_PHOTO_ACTIVITY_REQUEST_CODE)
    }

    private fun openCamera() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED ||
                ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA, Manifest.permission.WRITE_EXTERNAL_STORAGE), CAMERA_PERMISSION_REQUEST_CODE)
            return
        }
        val takePictureIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
        if (takePictureIntent.resolveActivity(packageManager) != null) {
            val timeStamp = SDF.format(Date())
            val imageFileName = "$timeStamp.jpg"
            val storageDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES)
            pictureImagePath = storageDir.absolutePath + "/" + imageFileName
            val file = File(pictureImagePath)
            val outputFileUri = Uri.fromFile(file)
            val builder = StrictMode.VmPolicy.Builder()
            StrictMode.setVmPolicy(builder.build())
            takePictureIntent.putExtra(MediaStore.EXTRA_OUTPUT, outputFileUri)
            startActivityForResult(takePictureIntent, TAKE_PHOTO_ACTIVITY_REQUEST_CODE)
        }
    }

    @Synchronized
    public override fun onStart() {
        LOGGER.d("onStart $this")
        super.onStart()
    }

    @Synchronized
    public override fun onResume() {
        LOGGER.d("onResume $this")
        super.onResume()

        handlerThread = HandlerThread("inference")
        handlerThread?.start()
        handler = Handler(handlerThread?.looper)
    }

    @Synchronized
    public override fun onPause() {
        LOGGER.d("onPause $this")

        handlerThread?.quitSafely()
        try {
            handlerThread?.join()
            handlerThread = null
            handler = null
        } catch (e: InterruptedException) {
            LOGGER.e(e, "Exception!")
        }

        super.onPause()
    }

    @Synchronized
    public override fun onStop() {
        LOGGER.d("onStop $this")
        super.onStop()
    }

    @Synchronized
    public override fun onDestroy() {
        LOGGER.d("onDestroy $this")
        super.onDestroy()
    }

    @Synchronized
    private fun runInBackground(r: Runnable) {
        handler?.post(r)
    }

    private fun runSegmentation(bitmap: Bitmap) {
        clearResults()

        // Rotate target image to .up orientation to avoid potential orientation misalignment.
        /*guard let targetImage = bitmap.transformOrientationToUp() else {
            inferenceStatusLabel.text = "ERROR: Image orientation couldn't be fixed."
            return
        }*/

        // Cache the target image.
        this.targetImage = bitmap

        // Center-crop the target image if the user has enabled the option.

        val image = if (cropSwitch.isChecked) this.targetImage?.cropCenter() else targetImage

        // Cache the potentially cropped image as input to the segmentation model.
        segmentationInput = image

        // Show the potentially cropped image on screen.
        imageView.setImageBitmap(image)

        // Lock the crop switch while segmentation is running.
        cropSwitch.isEnabled = false

        runInBackground( Runnable {
            // Make sure that image segmentator is initialized.
            imageSegmentator.let {
                // Make sure that the image is ready before running segmentation.
                image.let {

                    // Run image segmentation.
                    segmentationResult = imageSegmentator.runSegmentation(image!!)

                    runOnUiThread {
                        cropSwitch.isEnabled = true

                        // Change to show segmentation overlay result
                        segmentedControl.setPosition(2, true)
                        onSegmentChanged()

                        // Show result metadata
                        showInferenceTime(segmentationResult!!)
                        showClassLegend(segmentationResult!!)

                        // Enable switching between different display mode: input, segmentation, overlay
                        segmentedControl.isEnabled = true
                    }
                }
            }
        })
    }

    private fun showClassLegend(segmentationResult: SegmentationResult) {
        val spannable = SpannableStringBuilder()
        segmentationResult.colorLegend.forEach { (className, color) ->

            // If the color legend is light, use black text font. If not, use white text font.
            val textColor = if (isLight(color)) Color.BLACK else Color.WHITE

            spannable.append(className)
            spannable.append(" ")

            spannable.setSpan(ForegroundColorSpan(textColor), spannable.length - className.length - 1, spannable.length - 1, Spannable.SPAN_EXCLUSIVE_EXCLUSIVE)
            spannable.setSpan(BackgroundColorSpan(0xFF.shl(24).or(color)), spannable.length - className.length - 1, spannable.length - 1, Spannable.SPAN_EXCLUSIVE_EXCLUSIVE)
        }
        legendLabel.setText(spannable, TextView.BufferType.SPANNABLE)
    }

    private fun isLight(intColor:Int, threshold: Float = 0.5f): Boolean {

        val red = Color.red(intColor)
        val green = Color.green(intColor)
        val blue = Color.blue(intColor)

        // Calculate color brightness according to Digital ITU BT.601.
        val brightness = red * 0.299f + green * 0.587f + blue * 0.114

        return brightness > threshold * 255.0f
    }

    private fun showInferenceTime(segmentationResult: SegmentationResult) {
        val timeString = """Preprocessing: ${segmentationResult.preprocessingTime}ms.
Model inference: ${segmentationResult.inferenceTime}ms.
Postprocessing: ${segmentationResult.postProcessingTime}ms.
Visualization: ${segmentationResult.visualizationTime}ms."""
        inferenceStatusLabel.text = timeString
    }

    private fun onSegmentChanged() {
        when(segmentedControl.position) {
            0 -> imageView.setImageBitmap(segmentationInput)
            1 -> imageView.setImageBitmap(segmentationResult?.resultImage)
            2 -> imageView.setImageBitmap(segmentationResult?.overlayImage)
        }
    }

    private fun clearResults() {
        inferenceStatusLabel.text = getString(R.string.inference_running)
        legendLabel.text = ""
        segmentedControl.isEnabled = false
        segmentedControl.setPosition(0, true)
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        if (requestCode == CAMERA_PERMISSION_REQUEST_CODE) {
            if (permissions.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED && grantResults[1] == PackageManager.PERMISSION_GRANTED) {
                openCamera()
                return
            }
        } else if (requestCode == STORAGE_PERMISSION_REQUEST_CODE) {
            if (permissions.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                openGallery()
                return
            }
            return
        }
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {

        if (requestCode == TAKE_PHOTO_ACTIVITY_REQUEST_CODE) {
            if (resultCode != Activity.RESULT_OK) {
               return
            }
            val takenPhoto = File(pictureImagePath)
            if (takenPhoto.exists()) {
                val bitmap = readBitmapFromStorage(Uri.fromFile(takenPhoto))!!
                Handler().postDelayed({
                    runSegmentation(bitmap)
                }, ONE_SECOND_IN_MILLIS)
            }
            return
        } else if (requestCode == CHOOSE_PHOTO_ACTIVITY_REQUEST_CODE) {
            if (resultCode != Activity.RESULT_OK) {
                return
            }
            val bitmap = readBitmapFromStorage(data?.data!!)!!
            Handler().postDelayed({
                runSegmentation(bitmap)
            }, ONE_SECOND_IN_MILLIS)
            return
        }

        super.onActivityResult(requestCode, resultCode, data)
    }

    private fun readBitmapFromStorage(uri: Uri): Bitmap? {
        var options = BitmapFactory.Options().apply {
            inJustDecodeBounds = true
        }
        BitmapFactory.decodeStream(contentResolver.openInputStream(uri), null, options)
        val sampleSize = calculateInSampleSize(options, 1000, 1000)
        options = BitmapFactory.Options().apply {
            inJustDecodeBounds = false
            inSampleSize = sampleSize
        }
        val bitmap = BitmapFactory.decodeStream(contentResolver.openInputStream(uri), null, options)
        return bitmap?.transformUp(getOrientation(contentResolver.openInputStream(uri)))
    }

    private fun calculateInSampleSize(options: BitmapFactory.Options, reqWidth: Int, reqHeight: Int): Int {
        // Raw height and width of image
        val (height: Int, width: Int) = options.run { outHeight to outWidth }
        var inSampleSize = 1

        if (height > reqHeight || width > reqWidth) {

            val halfHeight: Int = height / 2
            val halfWidth: Int = width / 2

            // Calculate the largest inSampleSize value that is a power of 2 and keeps both
            // height and width larger than the requested height and width.
            while (halfHeight / inSampleSize >= reqHeight && halfWidth / inSampleSize >= reqWidth) {
                inSampleSize *= 2
            }
        }

        return inSampleSize
    }

    private fun getOrientation(inputStream: InputStream?): Int {
        val exifInterface: ExifInterface
        var orientation = 0
        try {
            exifInterface = ExifInterface(inputStream!!)
            orientation = exifInterface.getAttributeInt(
                    ExifInterface.TAG_ORIENTATION,
                    ExifInterface.ORIENTATION_UNDEFINED
            )
        } catch (e: IOException) {
            e.printStackTrace()
        }

        return orientation
    }
}
