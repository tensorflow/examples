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

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import android.util.Size
import java.io.Closeable
import java.util.PriorityQueue
import kotlin.math.min
import org.tensorflow.lite.DataType
import org.tensorflow.lite.InterpreterApi
import org.tensorflow.lite.InterpreterApi.Options.TfLiteRuntime
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.TensorProcessor
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp
import org.tensorflow.lite.support.image.ops.Rot90Op
import org.tensorflow.lite.support.label.TensorLabel
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer

/** Helper class used to communicate between our app and the TF image classification model */
class ImageClassificationHelper(context: Context) : Closeable {

  /** Abstraction object that wraps a classification output in an easy to parse way */
  data class Recognition(val id: String, val title: String, val confidence: Float)

  private val preprocessNormalizeOp = NormalizeOp(IMAGE_MEAN, IMAGE_STD)
  private val postprocessNormalizeOp = NormalizeOp(PROBABILITY_MEAN, PROBABILITY_STD)
  private val labels by lazy { FileUtil.loadLabels(context, LABELS_PATH) }
  private var tfInputBuffer = TensorImage(DataType.UINT8)
  private var tfImageProcessor: ImageProcessor? = null

  // Processor to apply post processing of the output probability
  private val probabilityProcessor = TensorProcessor.Builder().add(postprocessNormalizeOp).build()

  // Use TFLite in Play Services runtime by setting the option to FROM_SYSTEM_ONLY
  private val interpreterInitializer = lazy {
    val interpreterOption = InterpreterApi.Options().setRuntime(TfLiteRuntime.FROM_SYSTEM_ONLY)
    InterpreterApi.create(FileUtil.loadMappedFile(context, MODEL_PATH), interpreterOption)
  }
  // Only use interpreter after initialization finished in CameraActivity
  private val interpreter: InterpreterApi by interpreterInitializer
  private val tfInputSize by lazy {
    val inputIndex = 0
    val inputShape = interpreter.getInputTensor(inputIndex).shape()
    Size(inputShape[2], inputShape[1]) // Order of axis is: {1, height, width, 3}
  }

  // Output probability TensorBuffer
  private val outputProbabilityBuffer: TensorBuffer by lazy {
    val probabilityTensorIndex = 0
    val probabilityShape =
      interpreter.getOutputTensor(probabilityTensorIndex).shape() // {1, NUM_CLASSES}
    val probabilityDataType = interpreter.getOutputTensor(probabilityTensorIndex).dataType()
    TensorBuffer.createFixedSize(probabilityShape, probabilityDataType)
  }

  /** Classifies the input bitmapBuffer. */
  fun classify(bitmapBuffer: Bitmap, imageRotationDegrees: Int): List<Recognition> {
    // Loads the input bitmapBuffer
    tfInputBuffer = loadImage(bitmapBuffer, imageRotationDegrees)
    Log.d(TAG, "tensorSize: ${tfInputBuffer.width} x ${tfInputBuffer.height}")

    // Runs the inference call
    interpreter.run(tfInputBuffer.buffer, outputProbabilityBuffer.buffer.rewind())

    // Gets the map of label and probability
    val labeledProbability =
      TensorLabel(labels, probabilityProcessor.process(outputProbabilityBuffer)).mapWithFloatValue

    return getTopKProbability(labeledProbability)
  }

  /** Releases TFLite resources if initialized. */
  override fun close() {
    if (interpreterInitializer.isInitialized()) {
      interpreter.close()
    }
  }

  /** Loads input image, and applies preprocessing. */
  private fun loadImage(bitmapBuffer: Bitmap, imageRotationDegrees: Int): TensorImage {
    // Initializes preprocessor if null
    return (tfImageProcessor
        ?: run {
          val cropSize = minOf(bitmapBuffer.width, bitmapBuffer.height)
          ImageProcessor.Builder()
            .add(ResizeWithCropOrPadOp(cropSize, cropSize))
            .add(
              ResizeOp(
                tfInputSize.height,
                tfInputSize.width,
                ResizeOp.ResizeMethod.NEAREST_NEIGHBOR
              )
            )
            .add(Rot90Op(-imageRotationDegrees / 90))
            .add(preprocessNormalizeOp)
            .build()
            .also {
              tfImageProcessor = it
              Log.d(TAG, "tfImageProcessor initialized successfully. imageSize: $cropSize")
            }
        })
      .process(tfInputBuffer.apply { load(bitmapBuffer) })
  }

  /** Gets the top-k results. */
  private fun getTopKProbability(labelProb: Map<String, Float>): List<Recognition> {
    // Sort the recognition by confidence from high to low.
    val pq: PriorityQueue<Recognition> =
      PriorityQueue(MAX_RESULTS, compareByDescending<Recognition> { it.confidence })
    pq += labelProb.map { (label, prob) -> Recognition(label, label, prob) }
    return List(min(MAX_RESULTS, pq.size)) { pq.poll()!! }
  }

  companion object {
    private val TAG = ImageClassificationHelper::class.java.simpleName
    // Returns the top MAX_RESULTS recognitions
    private const val MAX_RESULTS = 10

    // ClassifierFloatEfficientNet model
    private const val MODEL_PATH = "efficientnet-lite0-fp32.tflite"
    private const val LABELS_PATH = "labels_without_background.txt"
    // Float model does not need dequantization in the post-processing. Setting mean and std as
    // 0.0f and 1.0f, respectively, to bypass the normalization
    private const val PROBABILITY_MEAN = 0.0f
    private const val PROBABILITY_STD = 1.0f
    private const val IMAGE_MEAN = 127.0f
    private const val IMAGE_STD = 128.0f
  }
}
