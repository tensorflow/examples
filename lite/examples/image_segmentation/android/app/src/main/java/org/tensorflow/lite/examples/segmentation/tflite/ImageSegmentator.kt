package org.tensorflow.lite.examples.segmentation.tflite

import android.app.Activity
import android.graphics.Bitmap
import android.os.SystemClock
import android.os.Trace
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import org.tensorflow.lite.examples.segmentation.env.Logger
import org.tensorflow.lite.examples.segmentation.ext.overlayWithImage
import java.nio.MappedByteBuffer
import java.util.*
import kotlin.collections.HashMap
import kotlin.math.min


class ImageSegmentator(activity: Activity) {

    private val LOGGER = Logger()

    private val IMAGE_MEAN = 127.5f
    private val IMAGE_STD = 127.5f

    private var tfliteModel: MappedByteBuffer

    /** An instance of the driver class to run model inference with Tensorflow Lite.  */
    private var tflite: Interpreter

    /** Options for configuring the Interpreter.  */
    private val tfliteOptions = Interpreter.Options()

    /** Labels corresponding to the output of the vision model.  */
    private val labels: List<String>

    /** Input image TensorBuffer.  */
    private var inputImageBuffer: TensorImage

    /** Output probability TensorBuffer.  */
    private val outputImageBuffer: TensorBuffer

    /// TF Lite Model's input and output shapes.
    private var batchSize: Int = 0
    private var inputImageWidth: Int = 0
    private var inputImageHeight: Int = 0
    private var inputPixelSize: Int = 0
    private var outputImageWidth: Int = 0
    private var outputImageHeight: Int = 0
    private var outputClassCount: Int = 0

    private var result: Array<Array<FloatArray>>

    init {
        tfliteModel = FileUtil.loadMappedFile(activity, getModelPath())
        tfliteOptions.setUseNNAPI(true)
        this.tflite = Interpreter(tfliteModel, tfliteOptions)
        labels = FileUtil.loadLabels(activity, getLabelPath())

        val inputShape = this.tflite.getInputTensor(0).shape() // {bacthSize, width, height, inputPixelSize}
        batchSize = inputShape[0]
        inputImageWidth = inputShape[1]
        inputImageHeight = inputShape[2]
        inputPixelSize = inputShape[3]

        val imageDataType = this.tflite.getInputTensor(0).dataType()

        val outputShape = this.tflite.getOutputTensor(0).shape() // {1, NUM_CLASSES}
        // Read output shape from model.
        outputImageWidth = outputShape[1]
        outputImageHeight = outputShape[2]
        outputClassCount = outputShape[3]

        inputImageBuffer = TensorImage(imageDataType)
        result = Array(outputImageWidth) { Array(outputImageHeight) { FloatArray(outputClassCount) } }

        val outputDataType = this.tflite.getOutputTensor(0).dataType()
        outputImageBuffer = TensorBuffer.createFixedSize(outputShape, outputDataType)

        LOGGER.d("Created a Tensorflow Lite Image Classifier.")
    }

    private fun getLabelPath(): String {
        return "labels.txt"
    }

    private fun getModelPath(): String {
        return "deeplabv3_257_mv_gpu.tflite"
    }

    private fun loadImage(bitmap: Bitmap): TensorImage {
        // Loads bitmap into a TensorImage.
        inputImageBuffer.load(bitmap.copy(bitmap.config, true))

        // Creates processor for the TensorImage.
        val cropSize = min(bitmap.width, bitmap.height)
        val imageProcessor = ImageProcessor.Builder()
                .add(ResizeWithCropOrPadOp(cropSize, cropSize))
                .add(ResizeOp(inputImageWidth, inputImageHeight, ResizeOp.ResizeMethod.BILINEAR))
                //.add(Rot90Op(numRoration))
                .add(NormalizeOp(IMAGE_MEAN, IMAGE_STD))
                .build()
        return imageProcessor.process(inputImageBuffer)
    }

    fun runSegmentation(bitmap: Bitmap) : SegmentationResult {
        // Logs this method so that it can be analyzed with systrace.
        Trace.beginSection("recognizeImage")

        Trace.beginSection("loadImage")
        val startTimeForLoadImage = SystemClock.uptimeMillis()
        inputImageBuffer = loadImage(bitmap)
        val endTimeForLoadImage = SystemClock.uptimeMillis()
        Trace.endSection()
        val preprocessingTime = (endTimeForLoadImage - startTimeForLoadImage)
        LOGGER.v("Timecost to load the image: " + (endTimeForLoadImage - startTimeForLoadImage))

        // Runs the inference call.
        Trace.beginSection("runInference")
        val startTimeForReference = SystemClock.uptimeMillis()
        tflite.run(inputImageBuffer.buffer, outputImageBuffer.buffer.rewind())
        val endTimeForReference = SystemClock.uptimeMillis()
        Trace.endSection()
        val inferenceTime = (endTimeForReference - startTimeForReference)
        LOGGER.v("Timecost to run model inference: " + (endTimeForReference - startTimeForReference))

        Trace.beginSection("parseOutput")
        val startTimeForPostProcess = SystemClock.uptimeMillis()
        val (map, pixels, classList) = parseOutputTensor(outputImageBuffer.floatArray)
        val endTimeForPostProcess = SystemClock.uptimeMillis()
        Trace.endSection()
        val postprocessingTime = (endTimeForPostProcess - startTimeForPostProcess)
        LOGGER.v("Timecost to run post process: " + (endTimeForPostProcess - startTimeForPostProcess))

        Trace.beginSection("visualize")
        val startTimeForVisualize = SystemClock.uptimeMillis()
        var resultImage = Bitmap.createBitmap(pixels, inputImageWidth, inputImageHeight, Bitmap.Config.ARGB_8888)
        resultImage = Bitmap.createScaledBitmap(resultImage, bitmap.width, bitmap.height, true)
        val overlayImage = bitmap.overlayWithImage(resultImage)
        val colorLegend = classListToColorLegend(classList)
        val endTimeForVisualize = SystemClock.uptimeMillis()
        Trace.endSection()
        val visualizationTime = (endTimeForVisualize - startTimeForVisualize)
        LOGGER.v("Timecost to run visualize: " + (endTimeForVisualize - startTimeForVisualize))

        return SegmentationResult(resultImage, overlayImage, preprocessingTime, inferenceTime, postprocessingTime, visualizationTime, colorLegend)
    }

    /// Look up the colors used to visualize the classes found in the image.
    private fun classListToColorLegend(classList: Set<Int>): Map<String, Int> {
        val colorLegend: HashMap<String, Int> = HashMap()
        classList.sorted().forEach { classIndex ->
            // Look up the color legend for the class.
            // Using modulo to reuse colors on segmentation model with large number of classes.
            val color = legendColorList[classIndex % legendColorList.size]
            colorLegend[labels[classIndex]] = color.toInt()
        }
        return colorLegend
    }

    private fun parseOutputTensor(outputTensor: FloatArray): OutputTensorResult {
        val segmentationMap = Array(outputImageWidth) { IntArray(outputImageHeight)}
        val segmentationImagePixels = IntArray(outputImageHeight * outputImageWidth)
        val classList = HashSet<Int>()

        var maxVal: Float
        var indexMax: Int

        // Looping through the output array
        for ( x in 0 until outputImageWidth){
            for (y in 0 until outputImageHeight) {
                // For each pixel, find the class that have the highest probability.
                maxVal = outputTensor[coordinateToIndex(x, y, 0)]
                indexMax = 0

                for (z in 1 until outputClassCount) {
                    val currentVal = outputTensor[coordinateToIndex(x, y, z)]
                    if (currentVal > maxVal) {
                        indexMax = z
                        maxVal = currentVal
                    }
                }

                // Store the most likely class to the output.
                segmentationMap[x][y] = indexMax
                classList.add(indexMax)

                // Lookup the color legend for the class.
                // Using modulo to reuse colors on segmentation model with large number of classes.
                val legendColor = legendColorList[indexMax % legendColorList.size]
                segmentationImagePixels[x * outputImageHeight + y] = legendColor.toInt()

            }
        }
        return OutputTensorResult(segmentationMap, segmentationImagePixels, classList)
    }

    /// Convert 3-dimension index (image_width x image_height x class_count) to 1-dimension index
    private fun coordinateToIndex(x: Int, y: Int, z: Int): Int {
        return x * outputImageHeight * outputClassCount + y * outputClassCount + z
    }


    companion object {

        /// List of colors to visualize segmentation result.
        val legendColorList = arrayOf(
                0xFFFFB300, // Vivid Yellow
                0xFF803E75, // Strong Purple
                0xFFFF6800, // Vivid Orange
                0xFFA6BDD7, // Very Light Blue
                0xFFC10020, // Vivid Red
                0xFFCEA262, // Grayish Yellow
                0xFF817066, // Medium Gray
                0xFF007D34, // Vivid Green
                0xFFF6768E, // Strong Purplish Pink
                0xFF00538A, // Strong Blue
                0xFFFF7A5C, // Strong Yellowish Pink
                0xFF53377A, // Strong Violet
                0xFFFF8E00, // Vivid Orange Yellow
                0xFFB32851, // Strong Purplish Red
                0xFFF4C800, // Vivid Greenish Yellow
                0xFF7F180D, // Strong Reddish Brown
                0xFF93AA00, // Vivid Yellowish Green
                0xFF593315, // Deep Yellowish Brown
                0xFFF13A13, // Vivid Reddish Orange
                0xFF232C16, // Dark Olive Green
                0xFF00A1C2 // Vivid Blue
        )
    }
}
