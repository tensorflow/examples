
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

package org.tensorflow.lite.examples.objectdetection

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.Rot90Op
import org.tensorflow.lite.task.core.BaseOptions
import org.tensorflow.lite.task.vision.detector.Detection
import org.tensorflow.lite.task.vision.detector.ObjectDetector

class ObjectDetectorHelper(
    var threshold: Float = DEFAULT_THRESHOLD,
    var numThreads: Int = DEFAULT_NUM_THREADS,
    var maxResults: Int = DEFAULT_MAX_RESULTS,
    var currentDelegate: Int = DELEGATE_CPU,
    var currentModel: Int = MODEL_EFFICIENTDETV0,
    val context: Context,
    val objectDetectorListener: DetectorListener?
) {

    companion object {
        private const val TAG = "ObjectDetectorHelper"

        // Detection configuration
        const val DEFAULT_THRESHOLD = 0.45f
        const val DEFAULT_NUM_THREADS = 4
        const val DEFAULT_MAX_RESULTS = 5

        // Hardware delegates
        const val DELEGATE_CPU = 0
        const val DELEGATE_GPU = 1
        const val DELEGATE_NNAPI = 2

        // Model identifiers
        const val MODEL_MOBILENETV1 = 0
        const val MODEL_EFFICIENTDETV0 = 1
        const val MODEL_EFFICIENTDETV1 = 2
        const val MODEL_EFFICIENTDETV2 = 3
    }

    private var objectDetector: ObjectDetector? = null

    init {
        setupObjectDetector()
    }

    /**
     * Releases the current detector reference.
     * Call setupObjectDetector() afterwards to initialize it again.
     */
    fun clearObjectDetector() {
        objectDetector = null
    }

    /**
     * Creates the detector using the selected model, confidence threshold,
     * result limit, thread count and hardware delegate.
     */
    @Synchronized
    fun setupObjectDetector() {
        // Release the previous detector reference before changing configuration.
        objectDetector = null

        try {
            val optionsBuilder =
                ObjectDetector.ObjectDetectorOptions.builder()
                    .setScoreThreshold(threshold.coerceIn(0.0f, 1.0f))
                    .setMaxResults(maxResults.coerceAtLeast(1))

            val baseOptionsBuilder =
                BaseOptions.builder()
                    .setNumThreads(numThreads.coerceAtLeast(1))

            when (currentDelegate) {
                DELEGATE_CPU -> {
                    // CPU is the default and most compatible option.
                }

                DELEGATE_GPU -> {
                    val compatibilityList = CompatibilityList()

                    if (compatibilityList.isDelegateSupportedOnThisDevice) {
                        baseOptionsBuilder.useGpu()
                    } else {
                        objectDetectorListener?.onError(
                            "GPU is not supported on this device. Using CPU instead."
                        )
                        Log.w(TAG, "GPU delegate unsupported; using CPU.")
                    }
                }

                DELEGATE_NNAPI -> {
                    baseOptionsBuilder.useNnapi()
                }

                else -> {
                    Log.w(TAG, "Unknown delegate $currentDelegate; using CPU.")
                }
            }

            optionsBuilder.setBaseOptions(baseOptionsBuilder.build())

            val modelName = when (currentModel) {
                MODEL_MOBILENETV1 -> "mobilenetv1.tflite"
                MODEL_EFFICIENTDETV0 -> "efficientdet-lite0.tflite"
                MODEL_EFFICIENTDETV1 -> "efficientdet-lite1.tflite"
                MODEL_EFFICIENTDETV2 -> "efficientdet-lite2.tflite"
                else -> {
                    Log.w(TAG, "Unknown model $currentModel; using MobileNet V1.")
                    "mobilenetv1.tflite"
                }
            }

            objectDetector =
                ObjectDetector.createFromFileAndOptions(
                    context,
                    modelName,
                    optionsBuilder.build()
                )

            Log.i(
                TAG,
                "Detector initialized: model=$modelName, " +
                    "threshold=$threshold, threads=$numThreads, " +
                    "maxResults=$maxResults, delegate=$currentDelegate"
            )
        } catch (e: Exception) {
            objectDetector = null

            Log.e(
                TAG,
                "Failed to initialize object detector.",
                e
            )

            objectDetectorListener?.onError(
                "Could not initialize the object detector. " +
                    "Check the model file and device configuration."
            )
        }
    }

    /**
     * Runs object detection on a bitmap.
     *
     * imageRotation should be the camera rotation in degrees, normally
     * 0, 90, 180 or 270, as supplied by the existing camera pipeline.
     */
    fun detect(image: Bitmap, imageRotation: Int) {
        val detector = objectDetector ?: run {
            setupObjectDetector()
            objectDetector
        }

        if (detector == null) {
            Log.e(TAG, "Detection skipped because the detector is unavailable.")
            return
        }

        val startTime = SystemClock.elapsedRealtime()

        try {
            val rotation = ((imageRotation % 360) + 360) % 360

            val imageProcessor =
                ImageProcessor.Builder()
                    .add(Rot90Op(-rotation / 90))
                    .build()

            val tensorImage =
                imageProcessor.process(
                    TensorImage.fromBitmap(image)
                )

            val results = detector.detect(tensorImage)

            val inferenceTime =
                SystemClock.elapsedRealtime() - startTime

            objectDetectorListener?.onResults(
                results,
                inferenceTime,
                tensorImage.height,
                tensorImage.width
            )
        } catch (e: Exception) {
            Log.e(TAG, "Object detection failed.", e)

            objectDetectorListener?.onError(
                "Object detection failed. Please try again."
            )
        }
    }

    interface DetectorListener {
        fun onError(error: String)

        fun onResults(
            results: MutableList<Detection>?,
            inferenceTime: Long,
            imageHeight: Int,
            imageWidth: Int
        )
    }
}
