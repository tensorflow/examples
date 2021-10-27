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

package org.tensorflow.lite.examples.transfer.api;

import java.io.Closeable;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/** Represents a "partially" trainable model that is based on some other, base model. */
public final class TransferLearningModel implements Closeable {

  /**
   * Prediction for a single class produced by the model.
   */
  public static class Prediction {
    private final String className;
    private final float confidence;

    public Prediction(String className, float confidence) {
      this.className = className;
      this.confidence = confidence;
    }

    public String getClassName() {
      return className;
    }

    public float getConfidence() {
      return confidence;
    }
  }

  private static class TrainingSample {
    float[] bottleneck;
    float[] label;

    TrainingSample(float[] bottleneck, float[] label) {
      this.bottleneck = bottleneck;
      this.label = label;
    }
  }

  /**
   * Consumer interface for training loss.
   */
  public interface LossConsumer {
    void onLoss(int epoch, float loss);
  }

  // Setting this to a higher value allows to calculate bottlenecks for more samples while
  // adding them to the bottleneck collection is blocked by an active training thread.
  private static final int NUM_THREADS =
      Math.max(1, Runtime.getRuntime().availableProcessors() - 1);

  private final Map<String, Integer> classes;
  private final String[] classesByIdx;
  private final Map<String, float[]> oneHotEncodedClass;

  private LiteMultipleSignatureModel model;

  private final List<TrainingSample> trainingSamples = new ArrayList<>();

  // Where to store training inputs.
  private float[][] trainingBatchBottlenecks;
  private float[][] trainingBatchLabels;

  // Used to spawn background threads.
  private final ExecutorService executor = Executors.newFixedThreadPool(NUM_THREADS);

  // This lock guarantees that only one thread is performing training and inference at
  // any point in time. It also protects the sample collection from being modified while
  // in use by a training thread.
  private final Lock trainingInferenceLock = new ReentrantLock();

  // This lock guards access to trainable parameters.
  private final ReadWriteLock parameterLock = new ReentrantReadWriteLock();

  // Set to true when [close] has been called.
  private volatile boolean isTerminating = false;

  public TransferLearningModel(ModelLoader modelLoader, Collection<String> classes) {
    try {
      this.model =
          new LiteMultipleSignatureModel(
              modelLoader.loadMappedFile("model.tflite"), classes.size());
    } catch (IOException e) {
      throw new RuntimeException("Couldn't read underlying model for TransferLearningModel", e);
    }
    classesByIdx = classes.toArray(new String[0]);
    this.classes = new TreeMap<>();
    oneHotEncodedClass = new HashMap<>();
    for (int classIdx = 0; classIdx < classes.size(); classIdx++) {
      String className = classesByIdx[classIdx];
      this.classes.put(className, classIdx);
      oneHotEncodedClass.put(className, oneHotEncoding(classIdx));
    }
  }

  /**
   * Adds a new sample for training.
   *
   * <p>Sample bottleneck is generated in a background thread, which resolves the returned Future
   * when the bottleneck is added to training samples.
   *
   * @param image image RGB data.
   * @param className ground truth label for image.
   */
  public Future<Void> addSample(float[][][] image, String className) {
    checkNotTerminating();

    if (!classes.containsKey(className)) {
      throw new IllegalArgumentException(String.format(
          "Class \"%s\" is not one of the classes recognized by the model", className));
    }

    return executor.submit(
        () -> {
          if (Thread.interrupted()) {
            return null;
          }

          trainingInferenceLock.lockInterruptibly();
          try {
            float[] bottleneck = model.loadBottleneck(image);
            trainingSamples.add(new TrainingSample(bottleneck, oneHotEncodedClass.get(className)));
          } finally {
            trainingInferenceLock.unlock();
          }

          return null;
        });
  }

  /**
   * Trains the model on the previously added data samples.
   *
   * @param numEpochs number of epochs to train for.
   * @param lossConsumer callback to receive loss values, may be null.
   * @return future that is resolved when training is finished.
   */
  public Future<Void> train(int numEpochs, LossConsumer lossConsumer) {
    checkNotTerminating();
    int trainBatchSize = getTrainBatchSize();

    if (trainingSamples.size() < trainBatchSize) {
      throw new RuntimeException(
          String.format(
              "Too few samples to start training: need %d, got %d",
              trainBatchSize, trainingSamples.size()));
    }

    trainingBatchBottlenecks = new float[trainBatchSize][numBottleneckFeatures()];
    trainingBatchLabels = new float[trainBatchSize][this.classes.size()];

    return executor.submit(
        () -> {
          trainingInferenceLock.lock();
          try {
            epochLoop:
            for (int epoch = 0; epoch < numEpochs; epoch++) {
              float totalLoss = 0;
              int numBatchesProcessed = 0;

              for (List<TrainingSample> batch : trainingBatches(trainBatchSize)) {
                if (Thread.interrupted()) {
                  break epochLoop;
                }

                for (int sampleIdx = 0; sampleIdx < batch.size(); sampleIdx++) {
                  TrainingSample sample = batch.get(sampleIdx);
                  trainingBatchBottlenecks[sampleIdx] = sample.bottleneck;
                  trainingBatchLabels[sampleIdx] = sample.label;
                }

                float loss = this.model.runTraining(trainingBatchBottlenecks, trainingBatchLabels);
                totalLoss += loss;
                numBatchesProcessed++;
              }

              float avgLoss = totalLoss / numBatchesProcessed;
              if (lossConsumer != null) {
                lossConsumer.onLoss(epoch, avgLoss);
              }
            }

            return null;
          } finally {
            trainingInferenceLock.unlock();
          }
        });
  }

  /**
   * Runs model inference on a given image.
   *
   * @param image image RGB data.
   * @return predictions sorted by confidence decreasing. Can be null if model is terminating.
   */
  public Prediction[] predict(float[][][] image) {
    checkNotTerminating();
    trainingInferenceLock.lock();

    try {
      if (isTerminating) {
        return null;
      }

      float[] confidences;
      parameterLock.readLock().lock();
      try {
        confidences = this.model.runInference(image);
      } finally {
        parameterLock.readLock().unlock();
      }

      Prediction[] predictions = new Prediction[classes.size()];
      for (int classIdx = 0; classIdx < classes.size(); classIdx++) {
        predictions[classIdx] = new Prediction(classesByIdx[classIdx], confidences[classIdx]);
      }

      Arrays.sort(predictions, (a, b) -> -Float.compare(a.confidence, b.confidence));
      return predictions;
    } finally {
      trainingInferenceLock.unlock();
    }
  }

  private float[] oneHotEncoding(int classIdx) {
    float[] oneHot = new float[4];
    oneHot[classIdx] = 1;
    return oneHot;
  }

  /** Training model expected batch size. */
  public int getTrainBatchSize() {
    return Math.min(
        Math.max(/* at least one sample needed */ 1, trainingSamples.size()),
        model.getExpectedBatchSize());
  }

  /**
   * Constructs an iterator that iterates over training sample batches.
   *
   * @param trainBatchSize batch size for training.
   * @return iterator over batches.
   */
  private Iterable<List<TrainingSample>> trainingBatches(int trainBatchSize) {
    if (!trainingInferenceLock.tryLock()) {
      throw new RuntimeException("Thread calling trainingBatches() must hold the training lock");
    }
    trainingInferenceLock.unlock();

    Collections.shuffle(trainingSamples);
    return () ->
        new Iterator<List<TrainingSample>>() {
          private int nextIndex = 0;

          @Override
          public boolean hasNext() {
            return nextIndex < trainingSamples.size();
          }

          @Override
          public List<TrainingSample> next() {
            int fromIndex = nextIndex;
            int toIndex = nextIndex + trainBatchSize;
            nextIndex = toIndex;
            if (toIndex >= trainingSamples.size()) {
              // To keep batch size consistent, last batch may include some elements from the
              // next-to-last batch.
              return trainingSamples.subList(
                  trainingSamples.size() - trainBatchSize, trainingSamples.size());
            } else {
              return trainingSamples.subList(fromIndex, toIndex);
            }
          }
        };
  }

  private int numBottleneckFeatures() {
    return model.getNumBottleneckFeatures();
  }

  private void checkNotTerminating() {
    if (isTerminating) {
      throw new IllegalStateException("Cannot operate on terminating model");
    }
  }

  /**
   * Terminates all model operation safely. Will block until current inference request is finished
   * (if any).
   *
   * <p>Calling any other method on this object after [close] is not allowed.
   */
  @Override
  public void close() {
    isTerminating = true;
    executor.shutdownNow();

    // Make sure that all threads doing inference are finished.
    trainingInferenceLock.lock();

    try {
      boolean ok = executor.awaitTermination(5, TimeUnit.SECONDS);
      if (!ok) {
        throw new RuntimeException("Model thread pool failed to terminate");
      }

      this.model.close();
    } catch (InterruptedException e) {
      // no-op
    } finally {
      trainingInferenceLock.unlock();
    }
  }
}
