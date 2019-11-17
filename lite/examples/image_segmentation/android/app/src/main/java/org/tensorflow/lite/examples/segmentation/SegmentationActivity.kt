package org.tensorflow.lite.examples.segmentation

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.appcompat.app.AppCompatActivity

import android.os.Bundle
import android.widget.ImageView
import android.widget.TextView
import org.tensorflow.lite.examples.segmentation.ext.cropCenter

import org.tensorflow.lite.examples.segmentation.tflite.ImageSegmentator
import org.tensorflow.lite.examples.segmentation.tflite.SegmentationResult
import android.text.Spannable
import android.text.style.ForegroundColorSpan
import android.graphics.Color
import android.os.Handler
import android.os.HandlerThread
import android.text.SpannableStringBuilder
import android.text.style.BackgroundColorSpan
import androidx.appcompat.widget.SwitchCompat
import com.addisonelliott.segmentedbutton.SegmentedButtonGroup
import org.tensorflow.lite.examples.segmentation.env.Logger

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

        Handler().postDelayed({
            runSegmentation(BitmapFactory.decodeResource(resources, R.drawable.boy))
        }, 1000)
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
}
