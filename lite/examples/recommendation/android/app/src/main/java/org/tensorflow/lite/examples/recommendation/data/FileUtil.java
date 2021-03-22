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

package org.tensorflow.lite.examples.recommendation.data;

import static java.nio.charset.StandardCharsets.UTF_8;
import static java.util.stream.Collectors.joining;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.lang.reflect.Type;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import org.tensorflow.lite.examples.recommendation.Config;

/** FileUtil class to load data from asset files. */
public class FileUtil {

  private FileUtil() {}

  /** Load TF Lite model from asset file. */
  public static MappedByteBuffer loadModelFile(AssetManager assetManager, String modelPath)
      throws IOException {
    try (AssetFileDescriptor fileDescriptor = assetManager.openFd(modelPath);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor())) {
      FileChannel fileChannel = inputStream.getChannel();
      long startOffset = fileDescriptor.getStartOffset();
      long declaredLength = fileDescriptor.getDeclaredLength();
      return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }
  }

  /** Load candidates from asset file. */
  public static Collection<MovieItem> loadMovieList(
      AssetManager assetManager, String candidateListPath) throws IOException {
    String content = loadFileContent(assetManager, candidateListPath);
    Gson gson = new Gson();
    Type type = new TypeToken<Collection<MovieItem>>() {}.getType();
    return gson.fromJson(content, type);
  }

  public static List<String> loadGenreList(AssetManager assetManager, String genreListPath)
      throws IOException {
    String content = loadFileContent(assetManager, genreListPath);
    String[] lines = content.split(System.lineSeparator());
    return Arrays.asList(lines);
  }

  /** Load config from asset file. */
  public static Config loadConfig(AssetManager assetManager, String configPath) throws IOException {
    String content = loadFileContent(assetManager, configPath);
    Gson gson = new Gson();
    Type type = new TypeToken<Config>() {}.getType();
    return gson.fromJson(content, type);
  }

  /** Load file content from asset file. */
  @SuppressWarnings("AndroidJdkLibsChecker")
  private static String loadFileContent(AssetManager assetManager, String path) throws IOException {
    try (InputStream ins = assetManager.open(path);
        BufferedReader reader = new BufferedReader(new InputStreamReader(ins, UTF_8))) {
      return reader.lines().collect(joining(System.lineSeparator()));
    }
  }
}
