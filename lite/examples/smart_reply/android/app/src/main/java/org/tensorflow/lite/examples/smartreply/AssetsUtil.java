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
package org.tensorflow.lite.examples.smartreply;

import static android.content.res.AssetManager.ACCESS_BUFFER;
import static android.os.ParcelFileDescriptor.MODE_READ_ONLY;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.os.ParcelFileDescriptor;
import com.google.common.io.ByteStreams;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

/** Helper to load assets. */
public class AssetsUtil {

  private AssetsUtil() {}

  /**
   * Gets AssetFileDescriptor directly for given a path, or returns its copy by caching for the
   * compressed one.
   */
  public static AssetFileDescriptor getAssetFileDescriptorOrCached(
      Context context, String assetPath) throws IOException {
    try {
      return context.getAssets().openFd(assetPath);
    } catch (FileNotFoundException e) {
      // If it cannot read from asset file (probably compressed), try copying to cache folder and
      // reloading.
      File cacheFile = new File(context.getCacheDir(), assetPath);
      cacheFile.getParentFile().mkdirs();
      copyToCacheFile(context, assetPath, cacheFile);
      ParcelFileDescriptor cachedFd = ParcelFileDescriptor.open(cacheFile, MODE_READ_ONLY);
      return new AssetFileDescriptor(cachedFd, 0, cacheFile.length());
    }
  }

  private static void copyToCacheFile(Context context, String assetPath, File cacheFile)
      throws IOException {
    try (InputStream inputStream = context.getAssets().open(assetPath, ACCESS_BUFFER);
        FileOutputStream fileOutputStream = new FileOutputStream(cacheFile, false)) {
      ByteStreams.copy(inputStream, fileOutputStream);
    }
  }
}
