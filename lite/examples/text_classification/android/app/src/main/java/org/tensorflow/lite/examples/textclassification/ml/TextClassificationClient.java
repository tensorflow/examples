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

package org.tensorflow.lite.examples.textclassification.ml;

import android.content.Context;

import java.util.ArrayList;
import java.util.List;

/**
 * Load TfLite model and provide predictions with task api.
 */
public class TextClassificationClient {
    private static final String NLCLASSIFIER = "NLCLASSIFIER";
    private static final String BertNLCLASSIFIER = "BERTNLCLASSIFIER";

    private final Context context;
    private String api;

    NLClassifierClient nlClassifierClient;
    BertNLClassifierClient bertNLClassifierClient;

    public TextClassificationClient(Context context, String api) {
        this.context = context;
        this.api = api;

        nlClassifierClient = new NLClassifierClient(context);
        bertNLClassifierClient = new BertNLClassifierClient(context);
    }

    /**
     * Load TF Lite model.
     */
    public void load() {
        if (api.equals(NLCLASSIFIER)) {

            nlClassifierClient.load();
        } else if (api.equals(BertNLCLASSIFIER)) {

            bertNLClassifierClient.load();
        }

    }

    /**
     * Free up resources as the client is no longer needed.
     */
    public void unload() {
        if (api.equals(NLCLASSIFIER)) {

            nlClassifierClient.unload();
        } else if (api.equals(BertNLCLASSIFIER)) {

            bertNLClassifierClient.unload();
        }
    }

    /**
     * Classify an input string and returns the classification results.
     */
    public List<Result> classify(String text) {
        List<Result> results = new ArrayList<>();
        if (api.equals(NLCLASSIFIER)) {

            results = nlClassifierClient.classify(text);
        } else if (api.equals(BertNLCLASSIFIER)) {

            results = bertNLClassifierClient.classify(text);
        }
        return results;
    }
}
