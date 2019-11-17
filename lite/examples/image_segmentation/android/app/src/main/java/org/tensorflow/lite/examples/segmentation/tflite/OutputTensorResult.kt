package org.tensorflow.lite.examples.segmentation.tflite

data class OutputTensorResult(val map: Array<IntArray>,
                              val pixels: IntArray,
                              val classSet: Set<Int>)
