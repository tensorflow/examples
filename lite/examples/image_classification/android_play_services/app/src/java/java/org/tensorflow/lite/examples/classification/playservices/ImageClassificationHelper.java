/*
 * Copyright 2020 The TensorFlow Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.classification.playservices;

import static java.lang.Math.min;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;
import android.util.Size;
import java.io.Closeable;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.PriorityQueue;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.InterpreterApi;
import org.tensorflow.lite.InterpreterApi.Options.TfLiteRuntime;
import org.tensorflow.lite.Tensor;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.TensorOperator;
import org.tensorflow.lite.support.common.TensorProcessor;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp;
import org.tensorflow.lite.support.image.ops.Rot90Op;
import org.tensorflow.lite.support.label.TensorLabel;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

/** Helper class used to communicate between our app and the TF image classification model. */
class ImageClassificationHelper implements Closeable {

  private static final String TAG = "ImageClassification";
  // ClassifierFloatEfficientNet model
  private static final String MODEL_PATH = "efficientnet-lite0-fp32.tflite";
  private static final String LABELS_PATH = "labels_without_background.txt";
  // Float model does not need dequantization in the post-processing. Setting mean and std as 0.0f
  // and 1.0f, respectively, to bypass the normalization
  private static final float PROBABILITY_MEAN = 0.0f;
  private static final float PROBABILITY_STD = 1.0f;
  private static final float IMAGE_MEAN = 127.0f;
  private static final float IMAGE_STD = 128.0f;
  private static final TensorOperator PREPROCESS_NORMALIZE_OP =
      new NormalizeOp(IMAGE_MEAN, IMAGE_STD);
  private static final TensorOperator POSTPROCESS_NORMALIZE_OP =
      new NormalizeOp(PROBABILITY_MEAN, PROBABILITY_STD);
  /** Processor to apply post processing of the output probability. */
  private static final TensorProcessor PROBABILITY_PROCESSOR =
      new TensorProcessor.Builder().add(POSTPROCESS_NORMALIZE_OP).build();

  /** Abstraction object that wraps a classification output in an easy to parse way. */
  public static class Recognition {
    private final String title;
    private final Float confidence;

    public Recognition(String title, Float confidence) {
      this.title = title;
      this.confidence = confidence;
    }

    public String getTitle() {
      return title;
    }

    public Float getConfidence() {
      return confidence;
    }
  }

  /**
   * Factory method to create an instance of {@code ImageClassificationHelper}.
   *
   * @param maxResults the number of {@link Recognition} that will be returned when classifying
   */
  public static ImageClassificationHelper create(Context context, int maxResults)
      throws IOException {
    // Use TFLite in Play Services runtime by setting the option to FROM_SYSTEM_ONLY
    InterpreterApi.Options options =
        new InterpreterApi.Options().setRuntime(TfLiteRuntime.FROM_SYSTEM_ONLY);
    InterpreterApi interpreter =
        InterpreterApi.create(FileUtil.loadMappedFile(context, MODEL_PATH), options);

    int[] inputShape = interpreter.getInputTensor(/* inputIndex */ 0).shape();
    Size tfInputSize =
        new Size(inputShape[2], inputShape[1]); // Order of axis is: {1, height, width, 3}

    Tensor outputTensor = interpreter.getOutputTensor(/* probabilityTensorIndex */ 0);
    TensorBuffer outputProbabilityBuffer =
        TensorBuffer.createFixedSize(outputTensor.shape(), outputTensor.dataType());

    // Read labels from file
    List<String> labels = FileUtil.loadLabels(context, LABELS_PATH);

    return new ImageClassificationHelper(
        maxResults, labels, interpreter, tfInputSize, outputProbabilityBuffer);
  }

  // Return the top maxResults classification result
  private final int maxResults;
  private final List<String> labels;
  // Only use interpreter after initialization finished in CameraActivity
  private final InterpreterApi interpreter;
  private final Size tfInputSize;
  private final TensorBuffer outputProbabilityBuffer;
  private ImageProcessor tfImageProcessor;
  private TensorImage tfInputBuffer = new TensorImage(DataType.UINT8);

  private ImageClassificationHelper(
      int maxResults,
      List<String> labels,
      InterpreterApi interpreter,
      Size tfInputSize,
      TensorBuffer outputProbabilityBuffer) {
    this.maxResults = maxResults;
    this.labels = labels;
    this.interpreter = interpreter;
    this.tfInputSize = tfInputSize;
    this.outputProbabilityBuffer = outputProbabilityBuffer;
  }

  /** Returns the top {@link #maxResults} recognition result in the input {@code bitmapBuffer}. */
  public List<Recognition> classify(final Bitmap bitmapBuffer, int imageRotationDegrees) {
    // Loads the input bitmapBuffer
    tfInputBuffer = loadImage(bitmapBuffer, imageRotationDegrees);
    Log.d(TAG, "tensorSize: " + tfInputBuffer.getWidth() + " x " + tfInputBuffer.getHeight());

    // Runs the inference call
    interpreter.run(tfInputBuffer.getBuffer(), outputProbabilityBuffer.getBuffer().rewind());

    // Gets the map of label and probability
    Map<String, Float> labeledProbability =
        new TensorLabel(labels, PROBABILITY_PROCESSOR.process(outputProbabilityBuffer))
            .getMapWithFloatValue();

    return getTopKProbability(labeledProbability, maxResults);
  }

  /** Releases TFLite resources. */
  @Override
  public void close() {
    interpreter.close();
  }

  /** Loads input image, and applies preprocessing. */
  private TensorImage loadImage(final Bitmap bitmapBuffer, int imageRotationDegrees) {
    // Initializes preprocessor if null
    if (tfImageProcessor == null) {
      int cropSize = min(bitmapBuffer.getWidth(), bitmapBuffer.getHeight());
      tfImageProcessor =
          new ImageProcessor.Builder()
              .add(new ResizeWithCropOrPadOp(cropSize, cropSize))
              .add(
                  new ResizeOp(
                      tfInputSize.getHeight(),
                      tfInputSize.getWidth(),
                      ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
              .add(new Rot90Op(-imageRotationDegrees / 90))
              .add(PREPROCESS_NORMALIZE_OP)
              .build();
      Log.d(TAG, "tfImageProcessor initialized successfully. imageSize: " + cropSize);
    }
    tfInputBuffer.load(bitmapBuffer);
    return tfImageProcessor.process(tfInputBuffer);
  }

  /** Gets the top {@code maxResults} results. */
  private static List<Recognition> getTopKProbability(
      Map<String, Float> labelProb, int maxResults) {
    // Sorts the recognition by confidence from HIGH to LOW
    PriorityQueue<Recognition> priorityQueue =
        new PriorityQueue<>(
            maxResults,
            (Recognition a, Recognition b) -> Float.compare(b.getConfidence(), a.getConfidence()));

    for (Entry<String, Float> entry : labelProb.entrySet()) {
      priorityQueue.add(new Recognition(entry.getKey(), entry.getValue()));
    }
    List<Recognition> recognitions = new ArrayList<>();
    int recognitionsSize = min(priorityQueue.size(), maxResults);
    for (int i = 0; i < recognitionsSize; ++i) {
      recognitions.add(priorityQueue.poll());
    }
    return recognitions;
  }
}
