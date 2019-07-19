/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
package org.tensorflow.lite.examples.posenet

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.drawable.Drawable
import android.os.Bundle
import android.support.v4.content.res.ResourcesCompat
import android.support.v7.app.AppCompatActivity
import android.widget.ImageView
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.examples.posenet.lib.Posenet as Posenet

class MainActivity : AppCompatActivity() {

  // Instantiate an Interpreter
  private var interpreter: Interpreter? = null

  // Preload and memory map the model file, returning a MappedByteBuffer containing the model.
  fun loadModelFile(path: String): MappedByteBuffer {
    val fileDescriptor = assets.openFd(path)
    val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
    return inputStream.channel.map(
      FileChannel.MapMode.READ_ONLY, fileDescriptor.startOffset, fileDescriptor.declaredLength
    )
  }

  // Returns a resized bitmap of the drawable image.
  private fun drawableToBitmap(drawable: Drawable): Bitmap {
    val bitmap = Bitmap.createBitmap(257, 353, Bitmap.Config.ARGB_8888)
    val canvas = Canvas(bitmap)

    drawable.setBounds(0, 0, canvas.getWidth(), canvas.getHeight())

    drawable.draw(canvas)
    return bitmap
  }

  // Calls the Posenet library functions.
  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    setContentView(R.layout.activity_main)

    val sampleImageView = findViewById<ImageView>(R.id.image)
    val drawedImage = ResourcesCompat.getDrawable(getResources(), R.drawable.image, null)
    val imageBitmap = drawableToBitmap(drawedImage!!)
    sampleImageView.setImageBitmap(imageBitmap)

    interpreter = Interpreter(loadModelFile("posenet_model.tflite"))
    val posenet = Posenet()
    val person = posenet.estimateSinglePose(interpreter!!, imageBitmap)

    // Draw the keypoints over the image.
    val red = Paint()
    red.setColor(Color.RED)
    val size = 2.0f

    val mutableBitmap = imageBitmap.copy(Bitmap.Config.ARGB_8888, true)
    val canvas = Canvas(mutableBitmap)
    for (i in 0 until person.keyPoints.size) {
      canvas.drawCircle(
        person.keyPoints[i].position.x.toFloat(),
        person.keyPoints[i].position.y.toFloat(), size, red
      )
    }
    sampleImageView.setAdjustViewBounds(true)
    sampleImageView.setImageBitmap(mutableBitmap)
  }
}
