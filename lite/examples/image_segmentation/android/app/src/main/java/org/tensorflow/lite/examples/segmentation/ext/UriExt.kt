package org.tensorflow.lite.examples.segmentation.ext

import android.content.Context
import android.net.Uri
import androidx.exifinterface.media.ExifInterface
import java.io.IOException

fun Uri.getOrientation(context: Context): Int {
    val exifInterface: ExifInterface
    var orientation = 0
    try {
        exifInterface = ExifInterface(context.contentResolver.openInputStream(this)!!)
        orientation = exifInterface.getAttributeInt(
                ExifInterface.TAG_ORIENTATION,
                ExifInterface.ORIENTATION_UNDEFINED
        )
    } catch (e: IOException) {
        e.printStackTrace()
    }

    return orientation
}