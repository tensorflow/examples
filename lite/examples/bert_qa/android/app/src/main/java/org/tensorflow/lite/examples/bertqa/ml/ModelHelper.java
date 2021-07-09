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
package org.tensorflow.lite.examples.bertqa.ml;

import static com.google.common.base.Verify.verify;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.util.Log;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.HashMap;
import java.util.Map;
import org.tensorflow.lite.support.metadata.MetadataExtractor;

/** Helper to load TfLite model and dictionary. */
public class ModelHelper {
  private static final String TAG = "BertDemo";
  public static final String MODEL_PATH = "model.tflite";
  public static final String DIC_PATH = "vocab.txt";

  private ModelHelper() {}

  /** Load tflite model from context. */
  public static MappedByteBuffer loadModelFile(Context context) throws IOException {
    return loadModelFile(context.getAssets());
  }

  /** Load tflite model from assets. */
  public static MappedByteBuffer loadModelFile(AssetManager assetManager) throws IOException {
    try (AssetFileDescriptor fileDescriptor = assetManager.openFd(MODEL_PATH);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor())) {
      FileChannel fileChannel = inputStream.getChannel();
      long startOffset = fileDescriptor.getStartOffset();
      long declaredLength = fileDescriptor.getDeclaredLength();
      return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }
  }

  /** Extract dictionary from metadata. */
  public static Map<String, Integer> extractDictionary(MetadataExtractor metadataExtractor) {
    Map<String, Integer> dic = null;
    try {
      verify(metadataExtractor != null, "metadataExtractor can't be null.");
      dic = loadDictionaryFile(metadataExtractor.getAssociatedFile(DIC_PATH));
      Log.v(TAG, "Dictionary loaded.");
    } catch (IOException ex) {
      Log.e(TAG, ex.getMessage());
    }
    return dic;
  }

  /** Load dictionary from assets. */
  public static Map<String, Integer> loadDictionaryFile(InputStream inputStream)
      throws IOException {
    Map<String, Integer> dic = new HashMap<>();
    try (BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream))) {
      int index = 0;
      while (reader.ready()) {
        String key = reader.readLine();
        dic.put(key, index++);
      }
    }
    return dic;
  }
}
