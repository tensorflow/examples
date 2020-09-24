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

package org.tensorflow.lite.examples.textclassification.client;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.util.Log;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.metadata.MetadataExtractor;

/** Interface to load TfLite model and provide predictions. */
public class TextClassificationClient {
  private static final String TAG = "Interpreter";

  private static final int SENTENCE_LEN = 256; // The maximum length of an input sentence.
  // Simple delimiter to split words.
  private static final String SIMPLE_SPACE_OR_PUNCTUATION = " |\\,|\\.|\\!|\\?|\n";
  private static final String MODEL_PATH = "text_classification.tflite";
  /*
   * Reserved values in ImdbDataSet dic:
   * dic["<PAD>"] = 0      used for padding
   * dic["<START>"] = 1    mark for the start of a sentence
   * dic["<UNKNOWN>"] = 2  mark for unknown words (OOV)
   */
  private static final String START = "<START>";
  private static final String PAD = "<PAD>";
  private static final String UNKNOWN = "<UNKNOWN>";

  /** Number of results to show in the UI. */
  private static final int MAX_RESULTS = 3;

  private final Context context;
  private final Map<String, Integer> dic = new HashMap<>();
  private final List<String> labels = new ArrayList<>();
  private Interpreter tflite;

  public TextClassificationClient(Context context) {
    this.context = context;
  }

  /** Load the TF Lite model and dictionary so that the client can start classifying text. */
  public void load() {
    loadModel();
  }

  /** Load TF Lite model. */
  private synchronized void loadModel() {
    try {
      // Load the TF Lite model
      ByteBuffer buffer = loadModelFile(this.context.getAssets(), MODEL_PATH);
      tflite = new Interpreter(buffer);
      Log.v(TAG, "TFLite model loaded.");

      // Use metadata extractor to extract the dictionary and label files.
      MetadataExtractor metadataExtractor = new MetadataExtractor(buffer);

      // Extract and load the dictionary file.
      InputStream dictionaryFile = metadataExtractor.getAssociatedFile("vocab.txt");
      loadDictionaryFile(dictionaryFile);
      Log.v(TAG, "Dictionary loaded.");

      // Extract and load the label file.
      InputStream labelFile = metadataExtractor.getAssociatedFile("labels.txt");
      loadLabelFile(labelFile);
      Log.v(TAG, "Labels loaded.");

    } catch (IOException ex) {
      Log.e(TAG, "Error loading TF Lite model.\n", ex);
    }
  }

  /** Free up resources as the client is no longer needed. */
  public synchronized void unload() {
    tflite.close();
    dic.clear();
    labels.clear();
  }

  /** Classify an input string and returns the classification results. */
  public synchronized List<Result> classify(String text) {
    // Pre-prosessing.
    int[][] input = tokenizeInputText(text);

    // Run inference.
    Log.v(TAG, "Classifying text with TF Lite...");
    float[][] output = new float[1][labels.size()];
    tflite.run(input, output);

    // Find the best classifications.
    PriorityQueue<Result> pq =
        new PriorityQueue<>(
            MAX_RESULTS, (lhs, rhs) -> Float.compare(rhs.getConfidence(), lhs.getConfidence()));
    for (int i = 0; i < labels.size(); i++) {
      pq.add(new Result("" + i, labels.get(i), output[0][i]));
    }
    final ArrayList<Result> results = new ArrayList<>();
    while (!pq.isEmpty()) {
      results.add(pq.poll());
    }

    Collections.sort(results);
    // Return the probability of each class.
    return results;
  }

  /** Load TF Lite model from assets. */
  private static MappedByteBuffer loadModelFile(AssetManager assetManager, String modelPath)
      throws IOException {
    try (AssetFileDescriptor fileDescriptor = assetManager.openFd(modelPath);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor())) {
      FileChannel fileChannel = inputStream.getChannel();
      long startOffset = fileDescriptor.getStartOffset();
      long declaredLength = fileDescriptor.getDeclaredLength();
      return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }
  }

  /** Load dictionary from model file. */
  private void loadLabelFile(InputStream ins) throws IOException {
    BufferedReader reader = new BufferedReader(new InputStreamReader(ins));
    // Each line in the label file is a label.
    while (reader.ready()) {
      labels.add(reader.readLine());
    }
  }

  /** Load labels from model file. */
  private void loadDictionaryFile(InputStream ins) throws IOException {
    BufferedReader reader = new BufferedReader(new InputStreamReader(ins));
    // Each line in the dictionary has two columns.
    // First column is a word, and the second is the index of this word.
    while (reader.ready()) {
      List<String> line = Arrays.asList(reader.readLine().split(" "));
      if (line.size() < 2) {
        continue;
      }
      dic.put(line.get(0), Integer.parseInt(line.get(1)));
    }
  }

  /** Pre-prosessing: tokenize and map the input words into a float array. */
  int[][] tokenizeInputText(String text) {
    int[] tmp = new int[SENTENCE_LEN];
    List<String> array = Arrays.asList(text.split(SIMPLE_SPACE_OR_PUNCTUATION));

    int index = 0;
    // Prepend <START> if it is in vocabulary file.
    if (dic.containsKey(START)) {
      tmp[index++] = dic.get(START);
    }

    for (String word : array) {
      if (index >= SENTENCE_LEN) {
        break;
      }
      tmp[index++] = dic.containsKey(word) ? dic.get(word) : (int) dic.get(UNKNOWN);
    }
    // Padding and wrapping.
    Arrays.fill(tmp, index, SENTENCE_LEN - 1, (int) dic.get(PAD));
    int[][] ans = {tmp};
    return ans;
  }

  Map<String, Integer> getDic() {
    return this.dic;
  }

  Interpreter getTflite() {
    return this.tflite;
  }

  List<String> getLabels() {
    return this.labels;
  }
}
