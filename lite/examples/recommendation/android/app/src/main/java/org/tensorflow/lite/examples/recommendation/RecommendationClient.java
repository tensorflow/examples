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

package org.tensorflow.lite.examples.recommendation;

import android.content.Context;
import android.util.Log;
import androidx.annotation.WorkerThread;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.examples.recommendation.data.FileUtil;
import org.tensorflow.lite.examples.recommendation.data.MovieItem;

/** Interface to load TfLite model and provide recommendations. */
public class RecommendationClient {
  private static final String TAG = "RecommendationClient";

  private final Context context;
  private final Map<Integer, MovieItem> candidates = new HashMap<>();
  private final Config config;
  private Interpreter tflite;

  /** An immutable result returned by a RecommendationClient. */
  public static class Result {

    /** Predicted id. */
    public final int id;

    /** Recommended item. */
    public final MovieItem item;

    /** A sortable score for how good the result is relative to others. Higher should be better. */
    public final float confidence;

    public Result(final int id, final MovieItem item, final float confidence) {
      this.id = id;
      this.item = item;
      this.confidence = confidence;
    }

    @Override
    public String toString() {
      return String.format("[%d] confidence: %.3f, item: %s", id, confidence, item);
    }
  }

  public RecommendationClient(Context context, Config config) {
    this.context = context;
    this.config = config;
  }

  /** Load the TF Lite model and dictionary. */
  @WorkerThread
  public void load() {
    loadModel();
    loadCandidateList();
  }

  /** Load TF Lite model. */
  @WorkerThread
  private synchronized void loadModel() {
    try {
      ByteBuffer buffer = FileUtil.loadModelFile(this.context.getAssets(), config.modelPath);
      tflite = new Interpreter(buffer);
      Log.v(TAG, "TFLite model loaded.");
    } catch (IOException ex) {
      Log.e(TAG, ex.getMessage());
    }
  }

  /** Load recommendation candidate list. */
  @WorkerThread
  private synchronized void loadCandidateList() {
    try {
      Collection<MovieItem> collection =
          FileUtil.loadMovieList(this.context.getAssets(), config.movieListPath);
      for (MovieItem item : collection) {
        Log.d(TAG, String.format("Load candidate: %s", item));
        candidates.put(item.id, item);
      }
      Log.v(TAG, "Candidate list loaded.");
    } catch (IOException ex) {
      Log.e(TAG, ex.getMessage());
    }
  }

  /** Free up resources as the client is no longer needed. */
  @WorkerThread
  public synchronized void unload() {
    tflite.close();
    candidates.clear();
  }

  /** Given a list of selected items, preprocess to get tflite input. */
  @WorkerThread
  synchronized int[] preprocess(List<MovieItem> selectedMovies) {
    int[] inputContext = new int[config.inputLength];
    for (int i = 0; i < config.inputLength; i++) {
      if (i < selectedMovies.size()) {
        MovieItem item = selectedMovies.get(i);
        inputContext[i] = item.id;
      } else {
        // Padding input.
        inputContext[i] = config.pad;
      }
    }
    return inputContext;
  }

  /** Postprocess to gets results from tflite inference. */
  @WorkerThread
  synchronized List<Result> postprocess(
      int[] outputIds, float[] confidences, List<MovieItem> selectedMovies) {
    final ArrayList<Result> results = new ArrayList<>();

    // Add recommendation results. Filter null or contained items.
    for (int i = 0; i < outputIds.length; i++) {
      if (results.size() >= config.topK) {
        Log.v(TAG, String.format("Selected top K: %d. Ignore the rest.", config.topK));
        break;
      }

      int id = outputIds[i];
      MovieItem item = candidates.get(id);
      if (item == null) {
        Log.v(TAG, String.format("Inference output[%d]. Id: %s is null", i, id));
        continue;
      }
      if (selectedMovies.contains(item)) {
        Log.v(TAG, String.format("Inference output[%d]. Id: %s is contained", i, id));
        continue;
      }
      Result result = new Result(id, item, confidences[i]);
      results.add(result);
      Log.v(TAG, String.format("Inference output[%d]. Result: %s", i, result));
    }

    return results;
  }

  /** Given a list of selected items, and returns the recommendation results. */
  @WorkerThread
  public synchronized List<Result> recommend(List<MovieItem> selectedMovies) {
    Object[] inputs = new Object[] {preprocess(selectedMovies)};

    // Run inference.
    int[] outputIds = new int[config.outputLength];
    float[] confidences = new float[config.outputLength];
    Map<Integer, Object> outputs = new HashMap<>();
    outputs.put(config.outputIdsIndex, outputIds);
    outputs.put(config.outputScoresIndex, confidences);
    tflite.runForMultipleInputsOutputs(inputs, outputs);

    return postprocess(outputIds, confidences, selectedMovies);
  }

  Map<Integer, MovieItem> getCandidates() {
    return this.candidates;
  }

  Interpreter getTflite() {
    return this.tflite;
  }
}
