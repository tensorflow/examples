/*
 * Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

package org.tensorflow.lite.examples.detection;

import static com.google.common.truth.Truth.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.mockito.MockitoAnnotations.openMocks;
import static java.lang.Math.abs;
import static java.lang.Math.max;
import static java.lang.Math.min;

import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.RectF;
import android.media.Image;

import androidx.test.core.app.ApplicationProvider;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.tensorflow.lite.examples.detection.tflite.Detector;
import org.tensorflow.lite.examples.detection.tflite.Detector.Recognition;
import org.tensorflow.lite.examples.detection.tflite.TFLiteObjectDetectionAPIModel;
import org.tensorflow.lite.support.image.ColorSpaceType;

/**
 * Golden test for Object Detection Reference app.
 */
@RunWith(AndroidJUnit4.class)
public class DetectorTest {

  private static final int MODEL_INPUT_SIZE = 300;
  private static final boolean IS_MODEL_QUANTIZED = true;
  private static final String MODEL_FILE = "detect.tflite";
  private static final String LABELS_FILE = "labelmap.txt";
  private Detector detector;
  private final Context applicationContext = ApplicationProvider.getApplicationContext();

  @Before
  public void setUp() throws IOException {
    openMocks(this);
    detector =
        TFLiteObjectDetectionAPIModel.create(
            applicationContext,
            MODEL_FILE,
            LABELS_FILE,
            MODEL_INPUT_SIZE,
            IS_MODEL_QUANTIZED);
  }

  @Test
  public void detectionResultsShouldNotChange() throws Exception {
    Bitmap assetsBitmap = loadImage("table.jpg");
    final List<Recognition> results = detector.recognizeImage(mockMediaImageFromBitmap(assetsBitmap, ColorSpaceType.NV21), 0);
    final List<Recognition> expected = loadRecognitions("table_results.txt");

    for (Recognition target : expected) {
      // Find a matching result in results
      boolean matched = false;
      for (Recognition item : results) {
        if (item.getTitle().equals(target.getTitle())
            && matchBoundingBoxes(item.getLocation(), target.getLocation())
            && matchConfidence(item.getConfidence(), target.getConfidence())) {
          matched = true;
          break;
        }
      }
      assertThat(matched).isTrue();
    }
  }

  // Confidence tolerance: absolute 1%
  private static boolean matchConfidence(float a, float b) {
    return abs(a - b) < 0.01;
  }

  // Bounding Box tolerance: overlapped area > 90% of each one
  private static boolean matchBoundingBoxes(RectF a, RectF b) {
    float areaA = a.width() * a.height();
    float areaB = b.width() * b.height();

    RectF overlapped =
        new RectF(
            max(a.left, b.left), max(a.top, b.top), min(a.right, b.right), min(a.bottom, b.bottom));
    float overlappedArea = overlapped.width() * overlapped.height();
    return overlappedArea > 0.9 * areaA && overlappedArea > 0.9 * areaB;
  }

  private static Bitmap loadImage(String fileName) throws Exception {
    AssetManager assetManager =
        InstrumentationRegistry.getInstrumentation().getContext().getAssets();
    InputStream inputStream = assetManager.open(fileName);
    return BitmapFactory.decodeStream(inputStream);
  }

  // The format of result:
  // category bbox.left bbox.top bbox.right bbox.bottom confidence
  // ...
  // Example:
  // Apple 99 25 30 75 0.99
  // Banana 25 90 75 200 0.98
  // ...
  private static List<Recognition> loadRecognitions(String fileName) throws Exception {
    AssetManager assetManager =
        InstrumentationRegistry.getInstrumentation().getContext().getAssets();
    InputStream inputStream = assetManager.open(fileName);
    Scanner scanner = new Scanner(inputStream);
    List<Recognition> result = new ArrayList<>();
    while (scanner.hasNext()) {
      String category = scanner.next();
      category = category.replace('_', ' ');
      if (!scanner.hasNextFloat()) {
        break;
      }
      float left = scanner.nextFloat();
      float top = scanner.nextFloat();
      float right = scanner.nextFloat();
      float bottom = scanner.nextFloat();
      RectF boundingBox = new RectF(left, top, right, bottom);
      float confidence = scanner.nextFloat();
      Recognition recognition = new Recognition(null, category, confidence, boundingBox);
      result.add(recognition);
    }
    return result;
  }

  private static YuvPlaneInfo createYuvPlaneInfo(
      ColorSpaceType colorSpaceType, int width, int height) {
    int uIndex = 0;
    int vIndex = 0;
    int uvPixelStride = 0;
    int yBufferSize = width * height;
    int uvBufferSize = ((width + 1) / 2) * ((height + 1) / 2);
    int uvRowStride = 0;
    switch (colorSpaceType) {
      case NV12:
        uIndex = yBufferSize;
        vIndex = yBufferSize + 1;
        uvPixelStride = 2;
        uvRowStride = (width + 1) / 2 * 2;
        break;
      case NV21:
        vIndex = yBufferSize;
        uIndex = yBufferSize + 1;
        uvPixelStride = 2;
        uvRowStride = (width + 1) / 2 * 2;
        break;
      case YV12:
        vIndex = yBufferSize;
        uIndex = yBufferSize + uvBufferSize;
        uvPixelStride = 1;
        uvRowStride = (width + 1) / 2;
        break;
      case YV21:
        uIndex = yBufferSize;
        vIndex = yBufferSize + uvBufferSize;
        uvPixelStride = 1;
        uvRowStride = (width + 1) / 2;
        break;
      default:
        throw new IllegalArgumentException(
            "ColorSpaceType: " + colorSpaceType.name() + ", is unsupported.");
    }

    return YuvPlaneInfo.create(
        uIndex,
        vIndex,
        /*yRowStride=*/ width,
        uvRowStride,
        uvPixelStride,
        yBufferSize,
        uvBufferSize);
  }

  private static byte[] getYuvBytesFromBitmap(Bitmap bitmap, ColorSpaceType colorSpaceType) {
    int width = bitmap.getWidth();
    int height = bitmap.getHeight();
    int[] rgb = new int[width * height];
    bitmap.getPixels(rgb, 0, width, 0, 0, width, height);

    YuvPlaneInfo yuvPlaneInfo = createYuvPlaneInfo(colorSpaceType, width, height);

    byte[] yuv = new byte[yuvPlaneInfo.getYBufferSize() + yuvPlaneInfo.getUvBufferSize() * 2];
    int rgbIndex = 0;
    int yIndex = 0;
    int vIndex = yuvPlaneInfo.getVIndex();
    int uIndex = yuvPlaneInfo.getUIndex();
    int uvPixelStride = yuvPlaneInfo.getUvPixelStride();

    for (int j = 0; j < height; ++j) {
      for (int i = 0; i < width; ++i) {
        int r = (rgb[rgbIndex] >> 16) & 0xff;
        int g = (rgb[rgbIndex] >> 8) & 0xff;
        int b = rgb[rgbIndex] & 0xff;

        int y = (int) (0.299f * r + 0.587f * g + 0.114f * b);
        int v = (int) ((r - y) * 0.713f + 128);
        int u = (int) ((b - y) * 0.564f + 128);

        yuv[yIndex++] = (byte) max(0, min(255, y));
        byte uByte = (byte) max(0, min(255, u));
        byte vByte = (byte) max(0, min(255, v));

        if ((i & 0x01) == 0 && (j & 0x01) == 0) {
          yuv[vIndex] = vByte;
          yuv[uIndex] = uByte;
          vIndex += uvPixelStride;
          uIndex += uvPixelStride;
        }

        rgbIndex++;
      }
    }
    return yuv;
  }

  public static Image mockMediaImageFromBitmap(Bitmap bitmap, ColorSpaceType colorSpaceType) {
    // Converts the RGB Bitmap to YUV TensorBuffer
    byte[] yuv = getYuvBytesFromBitmap(bitmap, colorSpaceType);

    int width = bitmap.getWidth();
    int height = bitmap.getHeight();
    YuvPlaneInfo yuvPlaneInfo = createYuvPlaneInfo(colorSpaceType, width, height);

    ByteBuffer yuvBuffer =
        ByteBuffer.allocateDirect(
            yuvPlaneInfo.getYBufferSize() + yuvPlaneInfo.getUvBufferSize() * 2);
    yuvBuffer.put(yuv);
    yuvBuffer.rewind();
    ByteBuffer yPlane = yuvBuffer.slice();

    yuvBuffer.rewind();
    yuvBuffer.position(yuvPlaneInfo.getUIndex());
    ByteBuffer uPlane = yuvBuffer.slice();

    yuvBuffer.rewind();
    yuvBuffer.position(yuvPlaneInfo.getVIndex());
    ByteBuffer vPlane = yuvBuffer.slice();

    Image.Plane mockPlaneY = mock(Image.Plane.class);
    when(mockPlaneY.getBuffer()).thenReturn(yPlane);
    when(mockPlaneY.getRowStride()).thenReturn(yuvPlaneInfo.getYRowStride());
    Image.Plane mockPlaneU = mock(Image.Plane.class);
    when(mockPlaneU.getBuffer()).thenReturn(uPlane);
    when(mockPlaneU.getRowStride()).thenReturn(yuvPlaneInfo.getUvRowStride());
    when(mockPlaneU.getPixelStride()).thenReturn(yuvPlaneInfo.getUvPixelStride());
    Image.Plane mockPlaneV = mock(Image.Plane.class);
    when(mockPlaneV.getBuffer()).thenReturn(vPlane);
    when(mockPlaneV.getRowStride()).thenReturn(yuvPlaneInfo.getUvRowStride());
    when(mockPlaneV.getPixelStride()).thenReturn(yuvPlaneInfo.getUvPixelStride());

    Image imageMock = mock(Image.class);
    when(imageMock.getFormat()).thenReturn(ImageFormat.YUV_420_888);
    when(imageMock.getPlanes()).thenReturn(new Image.Plane[]{mockPlaneY, mockPlaneU, mockPlaneV});
    when(imageMock.getWidth()).thenReturn(width);
    when(imageMock.getHeight()).thenReturn(height);
    return imageMock;
  }
}
