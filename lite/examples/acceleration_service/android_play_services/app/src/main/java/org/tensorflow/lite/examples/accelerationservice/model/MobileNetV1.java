/*
 * Copyright 2023 The TensorFlow Authors
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

package org.tensorflow.lite.examples.accelerationservice.model;

import static java.lang.Math.min;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import com.google.android.gms.tflite.acceleration.Model;
import com.google.android.gms.tflite.acceleration.Model.ModelLocation;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.PriorityQueue;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.examples.accelerationservice.logger.Logger;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.TensorOperator;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp;

/** Helper class for interaction with the MobileNetV1 model asset. */
final class MobileNetV1 implements AssetModel {

  private static final float IMAGE_MEAN = 127.0f;
  private static final float IMAGE_STD = 128.0f;

  private static final TensorOperator PREPROCESS_NORMALIZE_OP =
      new NormalizeOp(IMAGE_MEAN, IMAGE_STD);

  private static final String MODEL_ID = "mobilenet_v1_1.0_224";
  private static final String MODEL_NAMESPACE = "tflite";
  private static final String MODEL_PATH = MODEL_ID + "." + MODEL_NAMESPACE;

  private static final int TEST_IMAGE_HEIGHT = 224;
  private static final int TEST_IMAGE_WIDTH = 224;
  private static final int OUTPUT_CLASSES = 1001;

  private static final int MODEL_OUTPUT_DIMENSIONS = product(1, OUTPUT_CLASSES);

  private static final String TEST_IMAGE = "grace_hopper_224.jpg";
  private static final int TEST_IMAGE_OUTPUT_CLASS = 653; // 653 == "military uniform"

  private final Model model;
  private final Logger logger;
  private final TensorImage tfInputBuffer = new TensorImage(DataType.UINT8);

  private final Context context;
  private ImageProcessor tfImageProcessor;

  public MobileNetV1(Context context, Logger logger) throws IOException {
    ByteBuffer modelBuffer = FileUtil.loadMappedFile(context, MODEL_PATH);
    this.context = context;
    this.logger = logger;
    this.model =
        new Model.Builder()
            .setModelId(MODEL_ID)
            .setModelNamespace(MODEL_NAMESPACE)
            .setModelLocation(ModelLocation.fromByteBuffer(modelBuffer))
            .build();
  }

  @Override
  public Model getModel() {
    return model;
  }

  @Override
  public int getBatchSize() {
    return 1;
  }

  @Override
  public Object[] getInputs() {
    Bitmap image = readImage(TEST_IMAGE);
    ByteBuffer imageBuffer = loadImage(image).getBuffer();
    return new Object[] {imageBuffer};
  }

  @Override
  public Map<Integer, Object> allocateOutputs() {
    Map<Integer, Object> outputs = new HashMap<>();
    outputs.put(0, FloatBuffer.allocate(MODEL_OUTPUT_DIMENSIONS));
    return outputs;
  }

  @Override
  public boolean validateBenchmarkOutputs(ByteBuffer[] outputs) {
    for (int i = 0; i < outputs.length; i++) {
      float[] predictions = toFloatArray(outputs[i].asFloatBuffer());
      if (!validatePredictions(predictions)) {
        return false;
      }
    }
    return true;
  }

  @Override
  public boolean validateInterpreterOutputs(Object[] outputs) {
    for (int i = 0; i < outputs.length; i++) {
      float[] predictions = ((FloatBuffer) outputs[i]).array();
      if (!validatePredictions(predictions)) {
        return false;
      }
    }
    return true;
  }

  private boolean validatePredictions(float[] predictions) {
    if (predictions.length != OUTPUT_CLASSES) {
      logger.info(
          "Output sizes do not match: Got "
              + predictions.length
              + "-  Expected: "
              + OUTPUT_CLASSES);
      return false;
    }
    List<Map.Entry<Integer, Float>> results = getTopKLabels(predictions, 10);
    for (Map.Entry<Integer, Float> result : results) {
      logger.info("Class index: " + result.getKey() + " Confidence: " + result.getValue());
    }
    return results.get(0).getKey() == TEST_IMAGE_OUTPUT_CLASS;
  }

  private Bitmap readImage(String path) {
    try (InputStream stream = context.getAssets().open(path)) {
      return BitmapFactory.decodeStream(stream);
    } catch (IOException e) {
      throw new IllegalStateException("Couldn't read image: ", e);
    }
  }

  private TensorImage loadImage(final Bitmap bitmapBuffer) {
    // Initializes preprocessor if null
    if (tfImageProcessor == null) {
      int cropSize = min(bitmapBuffer.getWidth(), bitmapBuffer.getHeight());
      tfImageProcessor =
          new ImageProcessor.Builder()
              .add(new ResizeWithCropOrPadOp(cropSize, cropSize))
              .add(
                  new ResizeOp(
                      TEST_IMAGE_HEIGHT, TEST_IMAGE_WIDTH, ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
              .add(PREPROCESS_NORMALIZE_OP)
              .build();
    }
    tfInputBuffer.load(bitmapBuffer);
    return tfImageProcessor.process(tfInputBuffer);
  }

  private static final int product(int... values) {
    int total = 1;
    for (int v : values) {
      total *= v;
    }
    return total;
  }

  private static float[] toFloatArray(FloatBuffer output) {
    float[] predictions = new float[output.remaining()];
    output.get(predictions);
    return predictions;
  }

  private static List<Map.Entry<Integer, Float>> getTopKLabels(float[] labels, int k) {
    PriorityQueue<Map.Entry<Integer, Float>> pq =
        new PriorityQueue<>(
            k,
            (o1, o2) ->
                // Intentionally reversed to put high confidence at the head of the queue.
                o2.getValue().compareTo(o1.getValue()));

    for (int i = 0; i < labels.length; ++i) {
      pq.add(new AbstractMap.SimpleEntry<>(i, labels[i]));
    }
    List<Map.Entry<Integer, Float>> topKLabels = new ArrayList<>();
    for (int i = 0; i < min(pq.size(), k); ++i) {
      topKLabels.add(Objects.requireNonNull(pq.poll()));
    }
    return topKLabels;
  }
}
