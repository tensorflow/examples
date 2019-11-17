package org.tensorflow.lite.examples.segmentation.ext

import android.graphics.*
import android.media.ExifInterface
import kotlin.math.floor
import kotlin.math.min


fun Bitmap.transformUp(orientation: Int): Bitmap {
    val matrix = Matrix()
    when (orientation) {
        ExifInterface.ORIENTATION_NORMAL -> return this
        ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> matrix.setScale(-1f, 1f)
        ExifInterface.ORIENTATION_ROTATE_180 -> matrix.setRotate(180f)
        ExifInterface.ORIENTATION_FLIP_VERTICAL -> {
            matrix.setRotate(180f)
            matrix.postScale(-1f, 1f)
        }
        ExifInterface.ORIENTATION_TRANSPOSE -> {
            matrix.setRotate(90f)
            matrix.postScale(-1f, 1f)
        }
        ExifInterface.ORIENTATION_ROTATE_90 -> matrix.setRotate(90f)
        ExifInterface.ORIENTATION_TRANSVERSE -> {
            matrix.setRotate(-90f)
            matrix.postScale(-1f, 1f)
        }
        ExifInterface.ORIENTATION_ROTATE_270 -> matrix.setRotate(-90f)
        else -> return this
    }
    try {
        val bmRotated = Bitmap.createBitmap(
                this,
                0,
                0,
                width,
                height,
                matrix,
                true
        )
        if (bmRotated != this)
            recycle()
        return bmRotated
    } catch (e: OutOfMemoryError) {
        e.printStackTrace()
        return this
    }
}

fun Bitmap.cropCenter(): Bitmap {

    val isPortrait = width > height
    val isLandscape = width < height
    val breadth = min(width, height)

    val src = Rect(
           if (isPortrait) floor((width- height) / 2f).toInt() else 0,
           if (isLandscape) floor((height - width) / 2f).toInt() else 0,
           if (isPortrait) floor((height + width) / 2f).toInt() else breadth,
           if (isLandscape) floor((width + height) / 2f).toInt() else breadth
    )

    val dest = Rect(0, 0, breadth, breadth)

    val cropped = Bitmap.createBitmap(breadth, breadth, this.config)
    Canvas(cropped).drawBitmap(this, src, dest,null)

    return cropped
}

fun Bitmap.overlayWithImage(bitmap: Bitmap, alpha:Float = 0.5f): Bitmap {
    val copyBitmap = copy(config, true)
    val paint = Paint(Paint.ANTI_ALIAS_FLAG)
    paint.alpha = (alpha * 255).toInt()
    Canvas(copyBitmap).drawBitmap(bitmap, Matrix(), paint)
    return copyBitmap
}