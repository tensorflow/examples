package org.tensorflow.lite.examples.textclassification.ml;

import android.content.Context;
import android.util.Log;

import org.tensorflow.lite.support.label.Category;
import org.tensorflow.lite.task.text.nlclassifier.BertNLClassifier;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class BertNLClassifierClient {
    private static final String TAG = "BertNLClassifierTaskApi";
    private static final String MODEL_PATH = "text_classification.tflite";

    private final Context context;

    BertNLClassifier classifier;

    public BertNLClassifierClient(Context context) {
        this.context = context;
    }

    /**
     * Load TF Lite model.
     */
    public void load() {
        try {
            classifier = BertNLClassifier.createFromFile(context, MODEL_PATH);
            Log.d(TAG, "load");
        } catch (IOException e) {
            Log.e(TAG, e.getMessage());
        }
    }

    /**
     * Free up resources as the client is no longer needed.
     */
    public void unload() {
        classifier.close();
        classifier = null;
        Log.d(TAG, "unload");
    }

    /**
     * Classify an input string and returns the classification results.
     */
    public List<Result> classify(String text) {
        List<Category> apiResults = classifier.classify(text);
        List<Result> results = new ArrayList<>(apiResults.size());
        for (int i = 0; i < apiResults.size(); i++) {
            Category category = apiResults.get(i);
            results.add(new Result("" + i, category.getLabel(), category.getScore()));
        }
        Log.d(TAG, "classify");
        Collections.sort(results);
        Log.d(TAG, results.toString());
        return results;
    }
}
