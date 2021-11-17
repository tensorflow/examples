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

package org.tensorflow.lite.examples.imagesegmentation.tflite

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.os.SystemClock
import android.util.Log
import kotlin.collections.HashMap
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.task.core.BaseOptions
import org.tensorflow.lite.task.vision.segmenter.ImageSegmenter
import org.tensorflow.lite.task.vision.segmenter.ImageSegmenter.ImageSegmenterOptions
import org.tensorflow.lite.task.vision.segmenter.Segmentation

/**
 * Class responsible to run the Image Segmentation model. more information about the DeepLab model
 * being used can be found here:
 * https://ai.googleblog.com/2018/03/semantic-image-segmentation-with.html
 * https://www.tensorflow.org/lite/models/segmentation/overview
 * https://github.com/tensorflow/models/tree/master/research/deeplab
 *
 * Label names: 'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
 * 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
 * 'sofa', 'train', 'tv'
 */
class ImageSegmentationModelExecutor(context: Context, private var useGPU: Boolean = false) {

  private val imageSegmenter: ImageSegmenter

  private var fullTimeExecutionTime = 0L
  private var imageSegmentationTime = 0L
  private var maskFlatteningTime = 0L

  init {
    val baseOptionsBuilder = BaseOptions.builder()
    if (useGPU) {
      baseOptionsBuilder.useGpu()
    }
    val options =
      ImageSegmenterOptions.builder()
        .setBaseOptions(baseOptionsBuilder.setNumThreads(NUM_THREADS).build())
        .build()
    imageSegmenter =
      ImageSegmenter.createFromFileAndOptions(context, IMAGE_SEGMENTATION_MODEL, options)
  }

  fun execute(inputImage: Bitmap): ModelExecutionResult {
    try {
      fullTimeExecutionTime = SystemClock.uptimeMillis()

      imageSegmentationTime = SystemClock.uptimeMillis()
      val tensorImage = TensorImage.fromBitmap(inputImage)
      val results = imageSegmenter.segment(tensorImage)
      imageSegmentationTime = SystemClock.uptimeMillis() - imageSegmentationTime
      Log.d(TAG, "Time to run the ImageSegmenter $imageSegmentationTime")

      maskFlatteningTime = SystemClock.uptimeMillis()
      val (maskBitmap, itemsFound) =
        createMaskBitmapAndLabels(results.get(0), inputImage.getWidth(), inputImage.getHeight())
      maskFlatteningTime = SystemClock.uptimeMillis() - maskFlatteningTime
      Log.d(TAG, "Time to create the mask and labels $maskFlatteningTime")

      fullTimeExecutionTime = SystemClock.uptimeMillis() - fullTimeExecutionTime
      Log.d(TAG, "Total time execution $fullTimeExecutionTime")

      return ModelExecutionResult(
        /*bitmapResult=*/ stackTwoBitmaps(maskBitmap, inputImage),
        /*bitmapOriginal=*/ inputImage,
        /*bitmapMaskOnly=*/ maskBitmap,
        formatExecutionLog(inputImage.getWidth(), inputImage.getHeight()),
        itemsFound
      )
    } catch (e: Exception) {
      val exceptionLog = "something went wrong: ${e.message}"
      Log.d(TAG, exceptionLog)

      val emptyBitmap =
        Bitmap.createBitmap(inputImage.getWidth(), inputImage.getHeight(), Bitmap.Config.ARGB_8888)
      return ModelExecutionResult(
        emptyBitmap,
        emptyBitmap,
        emptyBitmap,
        exceptionLog,
        HashMap<String, Int>()
      )
    }
  }

  private fun createMaskBitmapAndLabels(
    result: Segmentation,
    inputWidth: Int,
    inputHeight: Int
  ): Pair<Bitmap, Map<String, Int>> {
    // For the sake of this demo, change the alpha channel from 255 (completely opaque) to 128
    // (semi-transparent), because the maskBitmap will be stacked over the original image later.
    val coloredLabels = result.getColoredLabels()
    var colors = IntArray(coloredLabels.size)
    var cnt = 0
    for (coloredLabel in coloredLabels) {
      val rgb = coloredLabel.getArgb()
      colors[cnt++] = Color.argb(ALPHA_VALUE, Color.red(rgb), Color.green(rgb), Color.blue(rgb))
    }
    // Use completely transparent for the background color.
    colors[0] = Color.TRANSPARENT

    // Create the mask bitmap with colors and the set of detected labels.
    val maskTensor = result.getMasks().get(0)
    val maskArray = maskTensor.getBuffer().array()
    val pixels = IntArray(maskArray.size)
    val itemsFound = HashMap<String, Int>()
    for (i in maskArray.indices) {
      val color = colors[maskArray[i].toInt()]
      pixels[i] = color
      itemsFound.put(coloredLabels.get(maskArray[i].toInt()).getlabel(), color)
    }
    val maskBitmap =
      Bitmap.createBitmap(
        pixels,
        maskTensor.getWidth(),
        maskTensor.getHeight(),
        Bitmap.Config.ARGB_8888
      )
    // Scale the maskBitmap to the same size as the input image.
    return Pair(Bitmap.createScaledBitmap(maskBitmap, inputWidth, inputHeight, true), itemsFound)
  }

  private fun stackTwoBitmaps(foregrand: Bitmap, background: Bitmap): Bitmap {
    val mergedBitmap =
      Bitmap.createBitmap(foregrand.getWidth(), foregrand.getHeight(), foregrand.getConfig())
    val canvas = Canvas(mergedBitmap)
    canvas.drawBitmap(background, 0.0f, 0.0f, null)
    canvas.drawBitmap(foregrand, 0.0f, 0.0f, null)
    return mergedBitmap
  }

  private fun formatExecutionLog(imageWidth: Int, imageHeight: Int): String {
    val sb = StringBuilder()
    sb.append("Input Image Size: $imageWidth x $imageHeight\n")
    sb.append("GPU enabled: $useGPU\n")
    sb.append("Number of threads: $NUM_THREADS\n")
    sb.append("ImageSegmenter execution time: $imageSegmentationTime ms\n")
    sb.append("Mask creation time: $maskFlatteningTime ms\n")
    sb.append("Full execution time: $fullTimeExecutionTime ms\n")
    return sb.toString()
  }

  fun close() {
    imageSegmenter.close()
  }

  companion object {
    public const val TAG = "SegmentationTask"
    private const val NUM_THREADS = 4
    private const val IMAGE_SEGMENTATION_MODEL = "deeplabv3_257_mv_gpu.tflite"
    private const val ALPHA_VALUE = 128
  }
}
