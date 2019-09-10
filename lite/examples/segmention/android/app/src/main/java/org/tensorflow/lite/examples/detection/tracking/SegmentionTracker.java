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

package org.tensorflow.lite.examples.detection.tracking;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Paint.Cap;
import android.graphics.Paint.Join;
import android.graphics.Paint.Style;
import android.graphics.RectF;
import android.text.TextUtils;
import android.util.Pair;
import android.util.TypedValue;

import org.tensorflow.lite.examples.detection.env.BorderedText;
import org.tensorflow.lite.examples.detection.env.ImageUtils;
import org.tensorflow.lite.examples.detection.env.Logger;
import org.tensorflow.lite.examples.detection.tflite.Classifier.Recognition;

import java.util.LinkedList;
import java.util.List;
import java.util.Queue;

/** A tracker that handles non-max suppression and matches existing objects to new detections. */
public class SegmentionTracker {
  private static final float TEXT_SIZE_DIP = 18;
  private static final float MIN_SIZE = 16.0f;
  private static final int MIN_CNT = 50;
  private static final int[] COLORS = {
    Color.BLUE,
    Color.RED,
    Color.GREEN,
    Color.YELLOW,
    Color.CYAN,
    Color.MAGENTA,
    Color.WHITE,
    Color.parseColor("#55FF55"),
    Color.parseColor("#FFA500"),
    Color.parseColor("#FF8888"),
    Color.parseColor("#AAAAFF"),
    Color.parseColor("#FFFFAA"),
    Color.parseColor("#55AAAA"),
    Color.parseColor("#AA33AA"),
    Color.parseColor("#0D0068")
  };
  final List<Pair<String, float[]>> screenRects = new LinkedList<Pair<String, float[]>>();
  private final Logger logger = new Logger();
  private final Queue<Integer> availableColors = new LinkedList<Integer>();
  private final List<TrackedRecognition> trackedObjects = new LinkedList<TrackedRecognition>();
  private final Paint contourPaint = new Paint();
  private final float textSizePx;
  private final BorderedText borderedText;
  private Matrix frameToCanvasMatrix;
  private int frameWidth;
  private int frameHeight;
  private int sensorOrientation;

  public SegmentionTracker(final Context context) {
    for (final int color : COLORS) {
      availableColors.add(color);
    }

    contourPaint.setColor(Color.RED);
    contourPaint.setStyle(Style.STROKE);
    contourPaint.setStrokeWidth(10.0f);
    contourPaint.setStrokeCap(Cap.ROUND);
    contourPaint.setStrokeJoin(Join.ROUND);
    contourPaint.setStrokeMiter(100);

    textSizePx =
        TypedValue.applyDimension(
            TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, context.getResources().getDisplayMetrics());
    borderedText = new BorderedText(textSizePx);
  }

  public synchronized void setFrameConfiguration(
      final int width, final int height, final int sensorOrientation) {
    frameWidth = width;
    frameHeight = height;
    this.sensorOrientation = sensorOrientation;
  }

  public synchronized void drawDebug(final Canvas canvas) {
    final Paint textPaint = new Paint();
    textPaint.setColor(Color.WHITE);
    textPaint.setTextSize(60.0f);

    final Paint contourPaint = new Paint();
    contourPaint.setColor(Color.RED);
    contourPaint.setAlpha(200);
    contourPaint.setStyle(Style.STROKE);

    for (final Pair<String, float[]> detection : screenRects) {
      final float[] contours = detection.second;
      canvas.drawPoints(contours, contourPaint);
      canvas.drawText("" + detection.first, contours[0], contours[1], textPaint);
      borderedText.drawText(canvas, contours[0], contours[1], "" + detection.first);
    }
  }

  public synchronized void trackResults(final List<Recognition> results, final long timestamp) {
    logger.i("Processing %d results from %d", results.size(), timestamp);
    processResults(results);
  }

  private Matrix getFrameToCanvasMatrix() {
    return frameToCanvasMatrix;
  }

  public synchronized void draw(final Canvas canvas) {
    final boolean rotated = sensorOrientation % 180 == 90;
    final float multiplier =
        Math.min(
            canvas.getHeight() / (float) (rotated ? frameWidth : frameHeight),
            canvas.getWidth() / (float) (rotated ? frameHeight : frameWidth));
    frameToCanvasMatrix =
        ImageUtils.getTransformationMatrix(
            frameWidth,
            frameHeight,
            (int) (multiplier * (rotated ? frameHeight : frameWidth)),
            (int) (multiplier * (rotated ? frameWidth : frameHeight)),
            sensorOrientation,
            false);

    for (final TrackedRecognition recognition : trackedObjects) {
      if(recognition.points.length == 0)
        continue;

      final float[] trackedPos = new float[recognition.points.length];
      System.arraycopy(recognition.points, 0, trackedPos, 0, recognition.points.length);

      getFrameToCanvasMatrix().mapPoints(trackedPos);
      contourPaint.setColor(recognition.color);
      canvas.drawPoints(trackedPos, contourPaint);

      final String labelString = String.format("%s", recognition.title);
      //            borderedText.drawText(canvas, trackedPos.left + cornerSize, trackedPos.top,
      // labelString);
      borderedText.drawText(
          canvas, trackedPos[0], trackedPos[1], labelString + "%", contourPaint);
    }
  }

  private void processResults(final List<Recognition> results) {
    final List<Pair<String, Recognition>> rectsToTrack = new LinkedList<Pair<String, Recognition>>();

    screenRects.clear();
    final Matrix rgbFrameToScreen = new Matrix(getFrameToCanvasMatrix());

    for (final Recognition result : results) {
      if (result.getContoursPonits() == null) {
        continue;
      }
      final float[] detectionFramePoints = result.getContoursPonits();

      final float[] detectionScreenPoints = new float[detectionFramePoints.length];
      rgbFrameToScreen.mapPoints(detectionScreenPoints, detectionFramePoints);

      logger.v(
          "Result! Frame: " + result.getContoursPonits() + " mapped to screen:" + detectionScreenPoints);

      screenRects.add(new Pair<String, float[]>(result.getId(), detectionScreenPoints));

      rectsToTrack.add(new Pair<String, Recognition>(result.getId(), result));
    }

    if (rectsToTrack.isEmpty()) {
      logger.v("Nothing to track, aborting.");
      return;
    }

    trackedObjects.clear();
    for (final Pair<String, Recognition> potential : rectsToTrack) {
      final TrackedRecognition trackedRecognition = new TrackedRecognition();
      trackedRecognition.points = potential.second.getContoursPonits();
      trackedRecognition.title = potential.second.getTitle();
      int colorIdx = Integer.parseInt(potential.first)/COLORS.length;
      trackedRecognition.color = COLORS[colorIdx];
      trackedObjects.add(trackedRecognition);
    }
  }

  private static class TrackedRecognition {
    float[] points;
    int color;
    String title;
  }
}
