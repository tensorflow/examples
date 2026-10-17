
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

package org.tensorflow.lite.examples.objectdetection

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import androidx.core.content.ContextCompat
import java.util.LinkedList
import java.util.Locale
import kotlin.math.max
import kotlin.math.min
import org.tensorflow.lite.task.vision.detector.Detection

class OverlayView(
    context: Context?,
    attrs: AttributeSet?
) : View(context, attrs) {

    private var results: List<Detection> = LinkedList()

    private val boxPaint = Paint(Paint.ANTI_ALIAS_FLAG)
    private val textBackgroundPaint = Paint(Paint.ANTI_ALIAS_FLAG)
    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG)

    private var scaleFactor: Float = 1f
    private val bounds = Rect()

    // Layout settings
    private val boxStrokeWidth = 5f
    private val textSizeSp = 14f
    private val textPadding = 8f
    private val cornerRadius = 8f

    init {
        initPaints()
    }

    private fun initPaints() {
        val density = resources.displayMetrics.density
        val scaledTextSize = textSizeSp * density

        boxPaint.apply {
            color = ContextCompat.getColor(
                context ?: return@apply,
                R.color.bounding_box_color
            )
            style = Paint.Style.STROKE
            strokeWidth = boxStrokeWidth * density
        }

        textBackgroundPaint.apply {
            color = Color.BLACK
            alpha = 220
            style = Paint.Style.FILL
        }

        textPaint.apply {
            color = Color.WHITE
            style = Paint.Style.FILL
            textSize = scaledTextSize
            typeface = android.graphics.Typeface.create(
                android.graphics.Typeface.DEFAULT,
                android.graphics.Typeface.BOLD
            )
        }
    }

    /**
     * Clears all displayed detections.
     */
    fun clear() {
        results = emptyList()
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        if (results.isEmpty() || width <= 0 || height <= 0) {
            return
        }

        for (result in results) {
            val category = result.categories.maxByOrNull { it.score }
                ?: continue

            val label = category.label.ifBlank { "Unknown object" }
            val confidence = (category.score * 100f)
                .coerceIn(0f, 100f)

            val boundingBox = result.boundingBox

            val left = boundingBox.left * scaleFactor
            val top = boundingBox.top * scaleFactor
            val right = boundingBox.right * scaleFactor
            val bottom = boundingBox.bottom * scaleFactor

            val drawableRect = RectF(
                left,
                top,
                right,
                bottom
            )

            // Draw the bounding box.
            canvas.drawRoundRect(
                drawableRect,
                cornerRadius,
                cornerRadius,
                boxPaint
            )

            // Display label and confidence as a percentage.
            val drawableText = String.format(
                Locale.US,
                "%s %.0f%%",
                label,
                confidence
            )

            textPaint.getTextBounds(
                drawableText,
                0,
                drawableText.length,
                bounds
            )

            val textWidth = textPaint.measureText(drawableText)
            val textHeight = bounds.height().toFloat()

            val backgroundWidth = textWidth + textPadding * 2
            val backgroundHeight = textHeight + textPadding * 2

            // Prefer placing the label above the bounding box.
            // If there is insufficient room, place it inside the box.
            val backgroundTop = if (top >= backgroundHeight) {
                top - backgroundHeight
            } else {
                top
            }

            val backgroundLeft = left.coerceAtLeast(0f)

            val backgroundRect = RectF(
                backgroundLeft,
                backgroundTop,
                min(backgroundLeft + backgroundWidth, width.toFloat()),
                min(backgroundTop + backgroundHeight, height.toFloat())
            )

            canvas.drawRoundRect(
                backgroundRect,
                cornerRadius,
                cornerRadius,
                textBackgroundPaint
            )

            // Keep the text inside the visible canvas.
            val textX = backgroundRect.left + textPadding
            val textBaseline = min(
                backgroundRect.bottom - textPadding,
                backgroundRect.top + textPadding + textHeight
            )

            canvas.drawText(
                drawableText,
                textX,
                textBaseline,
                textPaint
            )
        }
    }

    /**
     * Updates detections and scales bounding boxes to match the preview.
     */
    fun setResults(
        detectionResults: MutableList<Detection>,
        imageHeight: Int,
        imageWidth: Int
    ) {
        results = detectionResults.toList()

        if (
            imageWidth <= 0 ||
            imageHeight <= 0 ||
            width <= 0 ||
            height <= 0
        ) {
            scaleFactor = 1f
        } else {
            // Match the PreviewView's FILL_START scaling behavior.
            scaleFactor = max(
                width.toFloat() / imageWidth,
                height.toFloat() / imageHeight
            )
        }

        invalidate()
    }
}
