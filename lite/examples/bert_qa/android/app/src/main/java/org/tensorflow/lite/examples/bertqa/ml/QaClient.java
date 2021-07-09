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
import android.util.Log;
import androidx.annotation.WorkerThread;
import com.google.common.base.Joiner;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.metadata.MetadataExtractor;
import org.tensorflow.lite.support.metadata.schema.TensorMetadata;

/** Interface to load TfLite model and provide predictions. */
public class QaClient implements AutoCloseable {
  private static final String TAG = "BertDemo";

  private static final int MAX_ANS_LEN = 32;
  private static final int MAX_QUERY_LEN = 64;
  private static final int MAX_SEQ_LEN = 384;
  private static final boolean DO_LOWER_CASE = true;
  private static final int PREDICT_ANS_NUM = 5;
  private static final int NUM_LITE_THREADS = 4;

  private static final String IDS_TENSOR_NAME = "ids";
  private static final String MASK_TENSOR_NAME = "mask";
  private static final String SEGMENT_IDS_TENSOR_NAME = "segment_ids";
  private static final String END_LOGITS_TENSOR_NAME = "end_logits";
  private static final String START_LOGITS_TENSOR_NAME = "start_logits";

  // Need to shift 1 for outputs ([CLS]).
  private static final int OUTPUT_OFFSET = 1;

  private final Context context;
  private final Map<String, Integer> dic = new HashMap<>();
  private final FeatureConverter featureConverter;
  private Interpreter tflite;
  private MetadataExtractor metadataExtractor = null;

  private static final Joiner SPACE_JOINER = Joiner.on(" ");

  public QaClient(Context context) {
    this.context = context;
    this.featureConverter = new FeatureConverter(dic, DO_LOWER_CASE, MAX_QUERY_LEN, MAX_SEQ_LEN);
  }

  @WorkerThread
  public synchronized void loadModel() {
    try {
      ByteBuffer buffer = ModelHelper.loadModelFile(context);
      metadataExtractor = new MetadataExtractor(buffer);
      Map<String, Integer> loadedDic = ModelHelper.extractDictionary(metadataExtractor);
      verify(loadedDic != null, "dic can't be null.");
      dic.putAll(loadedDic);

      Interpreter.Options opt = new Interpreter.Options();
      opt.setNumThreads(NUM_LITE_THREADS);
      tflite = new Interpreter(buffer, opt);
      Log.v(TAG, "TFLite model loaded.");
    } catch (IOException ex) {
      Log.e(TAG, ex.getMessage());
    }
  }

  @WorkerThread
  public synchronized void unload() {
    close();
  }

  @Override
  public void close() {
    if (tflite != null) {
      tflite.close();
      tflite = null;
    }
    dic.clear();
  }


  /**
   * Input: Original content and query for the QA task. Later converted to Feature by
   * FeatureConverter. Output: A String[] array of answers and a float[] array of corresponding
   * logits.
   */
  @WorkerThread
  public synchronized List<QaAnswer> predict(String query, String content) {
    Log.v(TAG, "TFLite model: " + ModelHelper.MODEL_PATH + " running...");
    Log.v(TAG, "Convert Feature...");
    Feature feature = featureConverter.convert(query, content);

    Log.v(TAG, "Set inputs...");
    int[][] inputIds = new int[1][MAX_SEQ_LEN];
    int[][] inputMask = new int[1][MAX_SEQ_LEN];
    int[][] segmentIds = new int[1][MAX_SEQ_LEN];
    float[][] startLogits = new float[1][MAX_SEQ_LEN];
    float[][] endLogits = new float[1][MAX_SEQ_LEN];

    for (int j = 0; j < MAX_SEQ_LEN; j++) {
      inputIds[0][j] = feature.inputIds[j];
      inputMask[0][j] = feature.inputMask[j];
      segmentIds[0][j] = feature.segmentIds[j];
    }

    Object[] inputs = new Object[3];
    boolean useInputMetadata = false;
    if (metadataExtractor != null && metadataExtractor.getInputTensorCount() == 3) {
      // If metadata exists and the size of input tensors in metadata is 3, use metadata to treat
      // the tensor order. Since the order of input tensors can be different for different models,
      // set the inputs according to input tensor names.
      useInputMetadata = true;
      for (int i = 0; i < 3; i++) {
        TensorMetadata inputMetadata = metadataExtractor.getInputTensorMetadata(i);
        switch (inputMetadata.name()) {
          case IDS_TENSOR_NAME:
            inputs[i] = inputIds;
            break;
          case MASK_TENSOR_NAME:
            inputs[i] = inputMask;
            break;
          case SEGMENT_IDS_TENSOR_NAME:
            inputs[i] = segmentIds;
            break;
          default:
            Log.e(TAG, "Input name in metadata doesn't match the default input tensor names.");
            useInputMetadata = false;
        }
      }
    }
    if (!useInputMetadata) {
      // If metadata doesn't exists or doesn't contain the info, fail back to a hard-coded order.
      Log.v(TAG, "Use hard-coded order of input tensors.");
      inputs[0] = inputIds;
      inputs[1] = inputMask;
      inputs[2] = segmentIds;
    }

    Map<Integer, Object> output = new HashMap<>();
    // Hard-coded idx for output, maybe changed according to metadata below.
    int endLogitsIdx = 0;
    int startLogitsIdx = 1;
    boolean useOutputMetadata = false;
    if (metadataExtractor != null && metadataExtractor.getOutputTensorCount() == 2) {
      // If metadata exists and the size of output tensors in metadata is 2, use metadata to treat
      // the tensor order. Since the order of output tensors can be different for different models,
      // set the indexs of the outputs according to output tensor names.
      useOutputMetadata = true;
      for (int i = 0; i < 2; i++) {
        TensorMetadata outputMetadata = metadataExtractor.getOutputTensorMetadata(i);
        switch (outputMetadata.name()) {
          case END_LOGITS_TENSOR_NAME:
            endLogitsIdx = i;
            break;
          case START_LOGITS_TENSOR_NAME:
            startLogitsIdx = i;
            break;
          default:
            Log.e(TAG, "Output name in metadata doesn't match the default output tensor names.");
            useOutputMetadata = false;
        }
      }
    }
    if (!useOutputMetadata) {
      Log.v(TAG, "Use hard-coded order of output tensors.");
      endLogitsIdx = 0;
      startLogitsIdx = 1;
    }
    output.put(endLogitsIdx, endLogits);
    output.put(startLogitsIdx, startLogits);

    Log.v(TAG, "Run inference...");
    tflite.runForMultipleInputsOutputs(inputs, output);

    Log.v(TAG, "Convert answers...");
    List<QaAnswer> answers = getBestAnswers(startLogits[0], endLogits[0], feature);
    Log.v(TAG, "Finish.");
    return answers;
  }

  /** Find the Best N answers & logits from the logits array and input feature. */
  private synchronized List<QaAnswer> getBestAnswers(
      float[] startLogits, float[] endLogits, Feature feature) {
    // Model uses the closed interval [start, end] for indices.
    int[] startIndexes = getBestIndex(startLogits);
    int[] endIndexes = getBestIndex(endLogits);

    List<QaAnswer.Pos> origResults = new ArrayList<>();
    for (int start : startIndexes) {
      for (int end : endIndexes) {
        if (!feature.tokenToOrigMap.containsKey(start + OUTPUT_OFFSET)) {
          continue;
        }
        if (!feature.tokenToOrigMap.containsKey(end + OUTPUT_OFFSET)) {
          continue;
        }
        if (end < start) {
          continue;
        }
        int length = end - start + 1;
        if (length > MAX_ANS_LEN) {
          continue;
        }
        origResults.add(new QaAnswer.Pos(start, end, startLogits[start] + endLogits[end]));
      }
    }

    Collections.sort(origResults);

    List<QaAnswer> answers = new ArrayList<>();
    for (int i = 0; i < origResults.size(); i++) {
      if (i >= PREDICT_ANS_NUM) {
        break;
      }

      String convertedText;
      if (origResults.get(i).start > 0) {
        convertedText = convertBack(feature, origResults.get(i).start, origResults.get(i).end);
      } else {
        convertedText = "";
      }
      QaAnswer ans = new QaAnswer(convertedText, origResults.get(i));
      answers.add(ans);
    }
    return answers;
  }

  /** Get the n-best logits from a list of all the logits. */
  @WorkerThread
  private synchronized int[] getBestIndex(float[] logits) {
    List<QaAnswer.Pos> tmpList = new ArrayList<>();
    for (int i = 0; i < MAX_SEQ_LEN; i++) {
      tmpList.add(new QaAnswer.Pos(i, i, logits[i]));
    }
    Collections.sort(tmpList);

    int[] indexes = new int[PREDICT_ANS_NUM];
    for (int i = 0; i < PREDICT_ANS_NUM; i++) {
      indexes[i] = tmpList.get(i).start;
    }

    return indexes;
  }

  /** Convert the answer back to original text form. */
  @WorkerThread
  private static String convertBack(Feature feature, int start, int end) {
     // Shifted index is: index of logits + offset.
    int shiftedStart = start + OUTPUT_OFFSET;
    int shiftedEnd = end + OUTPUT_OFFSET;
    int startIndex = feature.tokenToOrigMap.get(shiftedStart);
    int endIndex = feature.tokenToOrigMap.get(shiftedEnd);
    // end + 1 for the closed interval.
    String ans = SPACE_JOINER.join(feature.origTokens.subList(startIndex, endIndex + 1));
    return ans;
  }
}
