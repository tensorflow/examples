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

package org.tensorflow.lite.examples.transfer;

import android.content.Context;
import android.os.Build;
import android.os.ConditionVariable;

import androidx.annotation.RequiresApi;

import java.io.Closeable;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;
import org.tensorflow.lite.examples.transfer.api.AssetModelLoader;
import org.tensorflow.lite.examples.transfer.api.TransferLearningModel;
import org.tensorflow.lite.examples.transfer.api.TransferLearningModel.LossConsumer;
import org.tensorflow.lite.examples.transfer.api.TransferLearningModel.Prediction;

/**
 * App-layer wrapper for {@link TransferLearningModel}.
 *
 * <p>This wrapper allows to run training continuously, using start/stop API, in contrast to
 * run-once API of {@link TransferLearningModel}.
 */
public class TransferLearningModelWrapper implements Closeable {
  public static final int IMAGE_SIZE = 224;

  private final TransferLearningModel model;

  private final ConditionVariable shouldTrain = new ConditionVariable();
  private volatile LossConsumer lossConsumer;
  private final String model_filename = "tflite-tl-trained.bin";

  private FileOutputStream modelOutputStream;

  private boolean fileExists(Context context, String filename) {
    File file = context.getFileStreamPath(filename);
    if(file == null || !file.exists()) {
      return false;
    }
    return true;
  }

  TransferLearningModelWrapper(Context context) {
    model =
        new TransferLearningModel(
                new AssetModelLoader(context, "model"), Arrays.asList("1", "2", "3", "4"));

    try {
      modelOutputStream = context.openFileOutput(model_filename, Context.MODE_PRIVATE);
      if (fileExists(context, model_filename)) {
        FileInputStream inputStream = context.openFileInput(model_filename);
        model.loadParameters(inputStream.getChannel());
      }
    } catch(IOException e){
      throw new RuntimeException("Exception occurred during model loading" + e, e.getCause());
    }

    new Thread(() -> {
      while (!Thread.interrupted()) {
        shouldTrain.block();
        try {
          model.train(1, lossConsumer).get();
        } catch (ExecutionException e) {
          throw new RuntimeException("Exception occurred during model training " + e, e.getCause());
        } catch (InterruptedException e) {
          // no-op
        }
      }
    }).start();
  }

  // This method is thread-safe.
  public Future<Void> addSample(float[] image, String className) {
    return model.addSample(image, className);
  }

  // This method is thread-safe, but blocking.
  public Prediction[] predict(float[] image) {
    return model.predict(image);
  }

  public int getTrainBatchSize() {
    return model.getTrainBatchSize();
  }

  /**
   * Start training the model continuously until {@link #disableTraining() disableTraining} is
   * called.
   *
   * @param lossConsumer callback that the loss values will be passed to.
   */
  public void enableTraining(LossConsumer lossConsumer) {
    this.lossConsumer = lossConsumer;
    shouldTrain.open();
  }

  /**
   * Stops training the model.
   */
  public void disableTraining() {
    shouldTrain.close();
    try {
      model.saveParameters(modelOutputStream.getChannel());
      modelOutputStream.flush();
    } catch (IOException e) {
      throw new RuntimeException("Exception occurred during model saving " + e, e.getCause());
    }
  }

  /** Frees all model resources and shuts down all background threads. */
  public void close() {
    model.close();
  }
}
