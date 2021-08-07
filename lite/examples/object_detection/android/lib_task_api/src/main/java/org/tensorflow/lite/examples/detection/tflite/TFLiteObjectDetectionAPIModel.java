/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package org.tensorflow.lite.examples.detection.tflite;

import android.content.Context;
import android.graphics.Matrix;
import android.graphics.Rect;
import android.media.Image;
import static java.lang.Math.min;
import android.os.Trace;

import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.util.ArrayList;
import java.util.List;

import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.task.core.vision.ImageProcessingOptions;
import org.tensorflow.lite.task.vision.detector.Detection;
import org.tensorflow.lite.task.vision.detector.ObjectDetector;
import org.tensorflow.lite.task.vision.detector.ObjectDetector.ObjectDetectorOptions;

/**
 * Wrapper for frozen detection models trained using the Tensorflow Object Detection API: -
 * https://github.com/tensorflow/models/tree/master/research/object_detection where you can find the
 * training code.
 *
 * <p>To use pretrained models in the API or convert to TF Lite models, please see docs for details:
 * -
 * https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md
 * -
 * https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md
 * -
 * https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tensorflowlite.md#running-our-model-on-android
 *
 * <p>For more information about Metadata and associated fields (eg: `labels.txt`), see <a
 * href="https://www.tensorflow.org/lite/convert/metadata#read_the_metadata_from_models">Read the
 * metadata from models</a>
 */

public class TFLiteObjectDetectionAPIModel implements Detector {
  private static final String TAG = "TFLiteObjectDetectionAPIModelWithTaskApi";

  /**
   * Only return this many results.
   */
  private static final int NUM_DETECTIONS = 10;

  private MappedByteBuffer modelBuffer;

  /**
   * An instance of the driver class to run model inference with Tensorflow Lite.
   */
  private ObjectDetector objectDetector;

  /**
   * Builder of the options used to config the ObjectDetector.
   */
  private ObjectDetectorOptions.Builder optionsBuilder;

  /**
   * Initializes a native TensorFlow session for classifying images.
   *
   * <p>{@code labelFilename}, {@code inputSize}, and {@code isQuantized}, are NOT required, but to
   * keep consistency with the implementation using the TFLite Interpreter Java API. See <a
   * href="https://github.com/tensorflow/examples/blob/master/lite/examples/object_detection/android/lib_interpreter/src/main/java/org/tensorflow/lite/examples/detection/tflite/TFLiteObjectDetectionAPIModel.java">lib_interpreter</a>.
   *
   * @param modelFilename The model file path relative to the assets folder
   * @param labelFilename The label file path relative to the assets folder
   * @param inputSize     The size of image input
   * @param isQuantized   Boolean representing model is quantized or not
   */
  public static Detector create(
          final Context context,
          final String modelFilename,
          final String labelFilename,
          final int inputSize,
          final boolean isQuantized)
          throws IOException {
    return new TFLiteObjectDetectionAPIModel(context, modelFilename);
  }

  private TFLiteObjectDetectionAPIModel(Context context, String modelFilename) throws IOException {
    modelBuffer = FileUtil.loadMappedFile(context, modelFilename);
    optionsBuilder = ObjectDetectorOptions.builder().setMaxResults(NUM_DETECTIONS);
    objectDetector = ObjectDetector.createFromBufferAndOptions(modelBuffer, optionsBuilder.build());
  }

  @Override
  public List<Recognition> recognizeImage(final Image image, int sensorOrientation) {
    // Log this method so that it can be analyzed with systrace.
    Trace.beginSection("recognizeImage");
    TensorImage inputImage = new TensorImage();
    inputImage.load(image);
    int width = image.getWidth();
    int height = image.getHeight();
    int cropSize = min(width, height);

    ImageProcessingOptions imageOptions =
            ImageProcessingOptions.builder()
                    .setOrientation(getOrientation(sensorOrientation))
                    // Set the ROI to the center of the image.
                    .setRoi(
                            new Rect(
                                    /*left=*/ (width - cropSize) / 2,
                                    /*top=*/ (height - cropSize) / 2,
                                    /*right=*/ (width + cropSize) / 2,
                                    /*bottom=*/ (height + cropSize) / 2))
                    .build();

    List<Detection> results = objectDetector.detect(inputImage, imageOptions);


    // Converts a list of {@link Detection} objects into a list of {@link Recognition} objects
    // to match the interface of other inference method, such as using the <a
    // href="https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection/android/lib_interpreter">TFLite
    // Java API.</a>.
    final ArrayList<Recognition> recognitions = new ArrayList<>();
    int cnt = 0;
    for (Detection detection : results) {
      recognitions.add(
              new Recognition(
                      "" + cnt++,
                      detection.getCategories().get(0).getLabel(),
                      detection.getCategories().get(0).getScore(),
                      detection.getBoundingBox()));
    }
    Trace.endSection(); // "recognizeImage"
    return recognitions;
  }

  /**
   * See http://jpegclub.org/exif_orientation.html for info
   *
   * @param cameraOrientation which is the degrees of the Image CameraX is providing
   * @return orientation to be used with ImageProcessingOptions
   */
  private static ImageProcessingOptions.Orientation getOrientation(int cameraOrientation) {
    ImageProcessingOptions.Orientation orientation;
    switch (cameraOrientation / 90) {
      case 1:
        orientation = ImageProcessingOptions.Orientation.LEFT_TOP;
        break;
      case 2:
        orientation = ImageProcessingOptions.Orientation.BOTTOM_LEFT;
        break;
      case 3:
        orientation = ImageProcessingOptions.Orientation.RIGHT_BOTTOM;
        break;
      default:
        orientation = ImageProcessingOptions.Orientation.TOP_RIGHT;
    }

    return orientation;
  }

  @Override
  public void enableStatLogging(final boolean logStats) {
  }

  @Override
  public String getStatString() {
    return "";
  }

  @Override
  public void close() {
    if (objectDetector != null) {
      objectDetector.close();
    }
  }

  @Override
  public void setNumThreads( int numThreads) {
    if (objectDetector != null) {
      optionsBuilder.setNumThreads(numThreads);
      recreateDetector();
    }
  }

  @Override
  public void setUseNNAPI(boolean isChecked) {
    throw new UnsupportedOperationException(
            "Manipulating the hardware accelerators is not allowed in the Task"
                    + " library currently. Only CPU is allowed.");
  }

  private void recreateDetector() {
    objectDetector.close();
    objectDetector = ObjectDetector.createFromBufferAndOptions(modelBuffer, optionsBuilder.build());

  }

  public static Matrix getTransformationMatrix(
          final int srcWidth,
          final int srcHeight,
          final int dstWidth,
          final int dstHeight,
          final int applyRotation,
          final boolean maintainAspectRatio) {
    final Matrix matrix = new Matrix();

    // Translate so center of image is at origin.
    matrix.postTranslate(-srcWidth / 2f, -srcHeight / 2f);

    if (applyRotation != 0) {
      // Rotate around origin.
      matrix.postRotate(applyRotation);
    }

    // Account for the already applied rotation, if any, and then determine how
    // much scaling is needed for each axis.
    final boolean transpose = (Math.abs(applyRotation) + 90) % 180 == 0;

    final int inWidth = transpose ? srcHeight : srcWidth;
    final int inHeight = transpose ? srcWidth : srcHeight;

    // Apply scaling if necessary.
    if (inWidth != dstWidth || inHeight != dstHeight) {
      final float scaleFactorX = dstWidth / (float) inWidth;
      final float scaleFactorY = dstHeight / (float) inHeight;

      if (maintainAspectRatio) {
        // Scale by minimum factor so that dst is filled completely while
        // maintaining the aspect ratio. Some image may fall off the edge.
        final float scaleFactor = Math.min(scaleFactorX, scaleFactorY);
        matrix.postScale(scaleFactor, scaleFactor);
      } else {
        // Scale exactly to fill dst from src.
        matrix.postScale(scaleFactorX, scaleFactorY);
      }
    }

    // Translate back from origin centered reference to destination frame.
    if (applyRotation == 90) {
      matrix.postTranslate(dstWidth / 3f, dstHeight / 2f);
    }else if(applyRotation == 0 || applyRotation == 180){
      matrix.postTranslate(dstWidth / 2f, dstHeight / 3f);
    }

    return matrix;
  }
}