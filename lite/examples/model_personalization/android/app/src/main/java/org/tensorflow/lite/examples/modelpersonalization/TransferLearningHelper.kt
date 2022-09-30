/*
 * Copyright 2022 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *             http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.modelpersonalization

import android.content.Context
import android.graphics.Bitmap
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.util.Log
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp
import org.tensorflow.lite.support.image.ops.Rot90Op
import org.tensorflow.lite.support.label.Category
import org.tensorflow.lite.support.label.TensorLabel
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.IOException
import java.nio.FloatBuffer
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.max
import kotlin.math.min


class TransferLearningHelper(
    var numThreads: Int = 2,
    val context: Context,
    val classifierListener: ClassifierListener?
) {

    private var interpreter: Interpreter? = null
    private val trainingSamples: MutableList<TrainingSample> = mutableListOf()
    private var executor: ExecutorService? = null

    //This lock guarantees that only one thread is performing training and
    //inference at any point in time.
    private val lock = Any()
    private var targetWidth: Int = 0
    private var targetHeight: Int = 0
    private val handler = Handler(Looper.getMainLooper())

    init {
        if (setupModelPersonalization()) {
            targetWidth = interpreter!!.getInputTensor(0).shape()[2]
            targetHeight = interpreter!!.getInputTensor(0).shape()[1]
        } else {
            classifierListener?.onError("TFLite failed to init.")
        }
    }

    fun close() {
        executor?.shutdownNow()
        executor = null
        interpreter = null
    }

    fun pauseTraining() {
        executor?.shutdownNow()
    }

    private fun setupModelPersonalization(): Boolean {
        val options = Interpreter.Options()
        options.numThreads = numThreads
        return try {
            val modelFile = FileUtil.loadMappedFile(context, "model.tflite")
            interpreter = Interpreter(modelFile, options)
            true
        } catch (e: IOException) {
            classifierListener?.onError(
                "Model personalization failed to " +
                        "initialize. See error logs for details"
            )
            Log.e(TAG, "TFLite failed to load model with error: " + e.message)
            false
        }
    }

    // Process input image and add the output into list samples which are
    // ready for training.
    fun addSample(image: Bitmap, className: String, rotation: Int) {
        synchronized(lock) {
            if (interpreter == null) {
                setupModelPersonalization()
            }
            processInputImage(image, rotation)?.let { tensorImage ->
                val bottleneck = loadBottleneck(tensorImage)
                trainingSamples.add(
                    TrainingSample(
                        bottleneck,
                        encoding(classes.getValue(className))
                    )
                )
            }
        }
    }

    // Start training process
    fun startTraining() {
        if (interpreter == null) {
            setupModelPersonalization()
        }

        // Create new thread for training process.
        executor = Executors.newSingleThreadExecutor()
        val trainBatchSize = getTrainBatchSize()

        if (trainingSamples.size < trainBatchSize) {
            throw RuntimeException(
                String.format(
                    "Too few samples to start training: need %d, got %d",
                    trainBatchSize, trainingSamples.size
                )
            )
        }

        executor?.execute {
            synchronized(lock) {
                var avgLoss: Float

                // Keep training until the helper pause or close.
                while (executor?.isShutdown == false) {
                    var totalLoss = 0f
                    var numBatchesProcessed = 0

                    // Shuffle training samples to reduce overfitting and
                    // variance.
                    trainingSamples.shuffle()

                    trainingBatches(trainBatchSize)
                        .forEach { trainingSamples ->
                            val trainingBatchBottlenecks =
                                MutableList(trainBatchSize) {
                                    FloatArray(
                                        BOTTLENECK_SIZE
                                    )
                                }

                            val trainingBatchLabels =
                                MutableList(trainBatchSize) {
                                    FloatArray(
                                        classes.size
                                    )
                                }

                            // Copy a training sample list into two different
                            // input training lists.
                            trainingSamples.forEachIndexed { index, trainingSample ->
                                trainingBatchBottlenecks[index] =
                                    trainingSample.bottleneck
                                trainingBatchLabels[index] =
                                    trainingSample.label
                            }

                            val loss = training(
                                trainingBatchBottlenecks,
                                trainingBatchLabels
                            )
                            totalLoss += loss
                            numBatchesProcessed++
                        }

                    // Calculate the average loss after training all batches.
                    avgLoss = totalLoss / numBatchesProcessed
                    handler.post {
                        classifierListener?.onLossResults(avgLoss)
                    }
                }
            }
        }
    }

    // Runs one training step with the given bottleneck batches and labels
    // and return the loss number.
    private fun training(
        bottlenecks: MutableList<FloatArray>,
        labels: MutableList<FloatArray>
    ): Float {
        val inputs: MutableMap<String, Any> = HashMap()
        inputs[TRAINING_INPUT_BOTTLENECK_KEY] = bottlenecks.toTypedArray()
        inputs[TRAINING_INPUT_LABELS_KEY] = labels.toTypedArray()

        val outputs: MutableMap<String, Any> = HashMap()
        val loss = FloatBuffer.allocate(1)
        outputs[TRAINING_OUTPUT_KEY] = loss

        interpreter?.runSignature(inputs, outputs, TRAINING_KEY)
        return loss.get(0)
    }

    // Invokes inference on the given image batches.
    fun classify(bitmap: Bitmap, rotation: Int) {
        processInputImage(bitmap, rotation)?.let { image ->
            synchronized(lock) {
                if (interpreter == null) {
                    setupModelPersonalization()
                }

                // Inference time is the difference between the system time at the start and finish of the
                // process
                var inferenceTime = SystemClock.uptimeMillis()

                val inputs: MutableMap<String, Any> = HashMap()
                inputs[INFERENCE_INPUT_KEY] = image.buffer

                val outputs: MutableMap<String, Any> = HashMap()
                val output = TensorBuffer.createFixedSize(
                    intArrayOf(1, 4),
                    DataType.FLOAT32
                )
                outputs[INFERENCE_OUTPUT_KEY] = output.buffer

                interpreter?.runSignature(inputs, outputs, INFERENCE_KEY)
                val tensorLabel = TensorLabel(classes.keys.toList(), output)
                val result = tensorLabel.categoryList

                inferenceTime = SystemClock.uptimeMillis() - inferenceTime

                classifierListener?.onResults(result, inferenceTime)
            }
        }
    }

    // Loads the bottleneck feature from the given image array.
    private fun loadBottleneck(image: TensorImage): FloatArray {
        val inputs: MutableMap<String, Any> = HashMap()
        inputs[LOAD_BOTTLENECK_INPUT_KEY] = image.buffer
        val outputs: MutableMap<String, Any> = HashMap()
        val bottleneck = Array(1) { FloatArray(BOTTLENECK_SIZE) }
        outputs[LOAD_BOTTLENECK_OUTPUT_KEY] = bottleneck
        interpreter?.runSignature(inputs, outputs, LOAD_BOTTLENECK_KEY)
        return bottleneck[0]
    }

    // Preprocess the image and convert it into a TensorImage for classification.
    private fun processInputImage(
        image: Bitmap,
        imageRotation: Int
    ): TensorImage? {
        val height = image.height
        val width = image.width
        val cropSize = min(height, width)
        val imageProcessor = ImageProcessor.Builder()
            .add(Rot90Op(-imageRotation / 90))
            .add(ResizeWithCropOrPadOp(cropSize, cropSize))
            .add(
                ResizeOp(
                    targetHeight,
                    targetWidth,
                    ResizeOp.ResizeMethod.BILINEAR
                )
            )
            .add(NormalizeOp(0f, 255f))
            .build()
        val tensorImage = TensorImage(DataType.FLOAT32)
        tensorImage.load(image)
        return imageProcessor.process(tensorImage)
    }

    // encode the classes name to float array
    private fun encoding(id: Int): FloatArray {
        val classEncoded = FloatArray(4) { 0f }
        classEncoded[id] = 1f
        return classEncoded
    }

    // Training model expected batch size.
    private fun getTrainBatchSize(): Int {
        return min(
            max( /* at least one sample needed */1, trainingSamples.size),
            EXPECTED_BATCH_SIZE
        )
    }

    // Constructs an iterator that iterates over training sample batches.
    private fun trainingBatches(trainBatchSize: Int): Iterator<List<TrainingSample>> {
        return object : Iterator<List<TrainingSample>> {
            private var nextIndex = 0

            override fun hasNext(): Boolean {
                return nextIndex < trainingSamples.size
            }

            override fun next(): List<TrainingSample> {
                val fromIndex = nextIndex
                val toIndex: Int = nextIndex + trainBatchSize
                nextIndex = toIndex
                return if (toIndex >= trainingSamples.size) {
                    // To keep batch size consistent, last batch may include some elements from the
                    // next-to-last batch.
                    trainingSamples.subList(
                        trainingSamples.size - trainBatchSize,
                        trainingSamples.size
                    )
                } else {
                    trainingSamples.subList(fromIndex, toIndex)
                }
            }
        }
    }

    interface ClassifierListener {
        fun onError(error: String)
        fun onResults(results: List<Category>?, inferenceTime: Long)
        fun onLossResults(lossNumber: Float)
    }

    companion object {
        const val CLASS_ONE = "1"
        const val CLASS_TWO = "2"
        const val CLASS_THREE = "3"
        const val CLASS_FOUR = "4"
        private val classes = mapOf(
            CLASS_ONE to 0,
            CLASS_TWO to 1,
            CLASS_THREE to 2,
            CLASS_FOUR to 3
        )
        private const val LOAD_BOTTLENECK_INPUT_KEY = "feature"
        private const val LOAD_BOTTLENECK_OUTPUT_KEY = "bottleneck"
        private const val LOAD_BOTTLENECK_KEY = "load"

        private const val TRAINING_INPUT_BOTTLENECK_KEY = "bottleneck"
        private const val TRAINING_INPUT_LABELS_KEY = "label"
        private const val TRAINING_OUTPUT_KEY = "loss"
        private const val TRAINING_KEY = "train"

        private const val INFERENCE_INPUT_KEY = "feature"
        private const val INFERENCE_OUTPUT_KEY = "output"
        private const val INFERENCE_KEY = "infer"

        private const val BOTTLENECK_SIZE = 1 * 7 * 7 * 1280
        private const val EXPECTED_BATCH_SIZE = 20
        private const val TAG = "ModelPersonalizationHelper"
    }

    data class TrainingSample(val bottleneck: FloatArray, val label: FloatArray)
}
