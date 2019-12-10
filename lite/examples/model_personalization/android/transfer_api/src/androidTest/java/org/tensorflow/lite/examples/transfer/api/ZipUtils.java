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

package org.tensorflow.lite.examples.transfer.api;

import android.content.Context;
import java.io.BufferedInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

class ZipUtils {
  private static final int BUFFER_SIZE = 65536;

  static Map<String, byte[]> readAllZipFiles(Context context, String pathToZip) throws IOException {
    Map<String, byte[]> result = new HashMap<>();
    ZipInputStream zin =
        new ZipInputStream(new BufferedInputStream(context.getAssets().open(pathToZip)));

    byte[] buffer = new byte[BUFFER_SIZE];

    ZipEntry zipEntry;
    while ((zipEntry = zin.getNextEntry()) != null) {
      ByteArrayOutputStream outputStream = new ByteArrayOutputStream();

      String filePath = zipEntry.getName();
      int readBytes;
      while ((readBytes = zin.read(buffer)) != -1) {
        outputStream.write(buffer, 0, readBytes);
      }

      result.put(filePath, outputStream.toByteArray());
    }

    return result;
  }
}
