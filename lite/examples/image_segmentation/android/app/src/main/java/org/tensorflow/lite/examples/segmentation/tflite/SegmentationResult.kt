package org.tensorflow.lite.examples.segmentation.tflite

import android.graphics.Bitmap

data class SegmentationResult(
        // Visualization of the segmentation result.
        val resultImage: Bitmap
        // Overlay the segmentation result on input image.
       ,val overlayImage: Bitmap
        // Processing time.
       ,val preprocessingTime: Long
       ,val inferenceTime: Long
       ,val postProcessingTime: Long
       ,val visualizationTime: Long
        // Dictionary of classes found in the image, and the color used to represent the class in
        // segmentation result visualization.
       ,val colorLegend: Map<String, Int>
)
