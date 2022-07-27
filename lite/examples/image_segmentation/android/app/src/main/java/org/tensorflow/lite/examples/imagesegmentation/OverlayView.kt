/*
 * Copyright 2022 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.imagesegmentation

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.util.AttributeSet
import android.view.View
import org.tensorflow.lite.task.vision.segmenter.ColoredLabel
import org.tensorflow.lite.task.vision.segmenter.Segmentation
import kotlin.math.max

class OverlayView(context: Context?, attrs: AttributeSet?) : View(context, attrs) {
    companion object {
        private const val ALPHA_COLOR = 128
    }

    private var scaleBitmap: Bitmap? = null
    private var listener: OverlayViewListener? = null

    fun setOnOverlayViewListener(listener: OverlayViewListener) {
        this.listener = listener
    }

    fun clear() {
        scaleBitmap = null
        invalidate()
    }

    override fun draw(canvas: Canvas) {
        super.draw(canvas)
        scaleBitmap?.let {
            canvas.drawBitmap(it, 0f, 0f, null)
        }
    }

    fun setResults(
        segmentResult: List<Segmentation>?,
        imageHeight: Int,
        imageWidth: Int,
    ) {
        if (segmentResult != null && segmentResult.isNotEmpty()) {
            val colorLabels = segmentResult[0].coloredLabels.mapIndexed { index, coloredLabel ->
                ColorLabel(
                    index,
                    coloredLabel.getlabel(),
                    coloredLabel.argb
                )
            }

            // Create the mask bitmap with colors and the set of detected labels.
            // We only need the first mask for this sample because we are using
            // the OutputType CATEGORY_MASK, which only provides a single mask.
            val maskTensor = segmentResult[0].masks[0]
            val maskArray = maskTensor.buffer.array()
            val pixels = IntArray(maskArray.size)

            for (i in maskArray.indices) {
                // Set isExist flag to true if any pixel contains this color.
                val colorLabel = colorLabels[maskArray[i].toInt()].apply {
                    isExist = true
                }
                val color = colorLabel.getColor()
                pixels[i] = color
            }

            val image = Bitmap.createBitmap(
                pixels,
                maskTensor.width,
                maskTensor.height,
                Bitmap.Config.ARGB_8888
            )

            // PreviewView is in FILL_START mode. So we need to scale up the bounding
            // box to match with the size that the captured images will be displayed.
            val scaleFactor = max(width * 1f / imageWidth, height * 1f / imageHeight)
            val scaleWidth = (imageWidth * scaleFactor).toInt()
            val scaleHeight = (imageHeight * scaleFactor).toInt()

            scaleBitmap = Bitmap.createScaledBitmap(image, scaleWidth, scaleHeight, false)
            listener?.onLabels(colorLabels.filter { it.isExist })
        }
    }

    interface OverlayViewListener {
        fun onLabels(colorLabels: List<ColorLabel>)
    }

    data class ColorLabel(
        val id: Int,
        val label: String,
        val rgbColor: Int,
        var isExist: Boolean = false
    ) {

        fun getColor(): Int {
            // Use completely transparent for the background color.
            return if (id == 0) Color.TRANSPARENT else Color.argb(
                ALPHA_COLOR,
                Color.red(rgbColor),
                Color.green(rgbColor),
                Color.blue(rgbColor)
            )
        }
    }
}
