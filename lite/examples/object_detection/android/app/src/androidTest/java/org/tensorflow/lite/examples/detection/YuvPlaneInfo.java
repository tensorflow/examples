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

import android.util.Log;

public class YuvPlaneInfo {

  int uIndexVar;
  int vIndexVar;
  int uvPixelStrideVar;
  int yBufferSizeVar;
  int uvBufferSizeVar;
  int uvRowStrideVar;
  int widthVar;

  public YuvPlaneInfo(int uIndex,
                      int vIndex,
                      int width,
                      int uvRowStride,
                      int uvPixelStride,
                      int yBufferSize,
                      int uvBufferSize) {
    this.uIndexVar = uIndex;
    this.vIndexVar = vIndex;
    this.widthVar = width;
    this.uvRowStrideVar = uvRowStride;
    this.uvPixelStrideVar = uvPixelStride;
    this.yBufferSizeVar = yBufferSize;
    this.uvBufferSizeVar = uvBufferSize;
  }

  public static YuvPlaneInfo create(int uIndex,
                                    int vIndex,
                                    int width,
                                    int uvRowStride,
                                    int uvPixelStride,
                                    int yBufferSize,
                                    int uvBufferSize) {
    return new YuvPlaneInfo(uIndex, vIndex, width, uvRowStride, uvPixelStride, yBufferSize, uvBufferSize);
  }

  public int getYBufferSize() {
    Log.v("YuvPlaneInfo", String.valueOf(yBufferSizeVar));
    return yBufferSizeVar;
  }

  public int getUvBufferSize() {
    Log.v("YuvPlaneInfo", String.valueOf(uvBufferSizeVar));
    return uvBufferSizeVar;
  }

  public int getVIndex() {
    return vIndexVar;
  }

  public int getUIndex() {
    return uIndexVar;
  }

  public int getUvPixelStride() {
    return uvPixelStrideVar;
  }

  public int getYRowStride() {
    return widthVar;
  }


  public int getUvRowStride() {
    return uvRowStrideVar;
  }

}
