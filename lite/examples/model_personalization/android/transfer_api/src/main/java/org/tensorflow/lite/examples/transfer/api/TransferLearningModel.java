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
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.GatheringByteChannel;
import java.nio.channels.ScatteringByteChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
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

/**
 * Represents a "partially" trainable model that is based on some other,
 * base model.
 */
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
    ByteBuffer bottleneck;
    String className;

    TrainingSample(ByteBuffer bottleneck, String className) {
      this.bottleneck = bottleneck;
      this.className = className;
    }
  }

  /**
   * Consumer interface for training loss.
   */
  public interface LossConsumer {
    void onLoss(int epoch, float loss);
  }

  private static final int FLOAT_BYTES = 4;

  // Setting this to a higher value allows to calculate bottlenecks for more samples while
  // adding them to the bottleneck collection is blocked by an active training thread.
  private static final int NUM_THREADS =
      Math.max(1, Runtime.getRuntime().availableProcessors() - 1);

  private final int[] bottleneckShape;

  private final Map<String, Integer> classes;
  private final String[] classesByIdx;

  private final LiteInitializeModel initializeModel;
  private final LiteBottleneckModel bottleneckModel;
  private final LiteTrainHeadModel trainHeadModel;
  private final LiteInferenceModel inferenceModel;
  private final LiteOptimizerModel optimizerModel;

  private final List<TrainingSample> trainingSamples = new ArrayList<>();

  private ByteBuffer[] modelParameters;

  // Where to store the optimizer outputs.
  private ByteBuffer[] nextModelParameters;

  private ByteBuffer[] optimizerState;

  // Where to store the updated optimizer state.
  private ByteBuffer[] nextOptimizerState;

  // Where to store training inputs.
  private final ByteBuffer trainingBatchBottlenecks;
  private final ByteBuffer trainingBatchClasses;

  // A zero-filled buffer of the same size as `trainingBatchClasses`.
  private final ByteBuffer zeroBatchClasses;

  // Where to store calculated gradients.
  private final ByteBuffer[] modelGradients;

  // Where to store bottlenecks produced during inference.
  private ByteBuffer inferenceBottleneck;

  // Used to spawn background threads.
  private final ExecutorService executor = Executors.newFixedThreadPool(NUM_THREADS);

  // This lock guarantees that only one thread is performing training at any point in time.
  // It also protects the sample collection from being modified while in use by a training
  // thread.
  private final Lock trainingLock = new ReentrantLock();

  // This lock guards access to trainable parameters.
  private final ReadWriteLock parameterLock = new ReentrantReadWriteLock();

  // This lock allows [close] method to assure that no threads are performing inference.
  private final Lock inferenceLock = new ReentrantLock();

  // Set to true when [close] has been called.
  private volatile boolean isTerminating = false;

  public TransferLearningModel(ModelLoader modelLoader, Collection<String> classes) {
    classesByIdx = classes.toArray(new String[0]);
    this.classes = new TreeMap<>();
    for (int classIdx = 0; classIdx < classes.size(); classIdx++) {
      this.classes.put(classesByIdx[classIdx], classIdx);
    }

    try {
      initializeModel = new LiteInitializeModel(modelLoader.loadInitializeModel());
      bottleneckModel = new LiteBottleneckModel(modelLoader.loadBaseModel());
      trainHeadModel = new LiteTrainHeadModel(modelLoader.loadTrainModel());
      inferenceModel = new LiteInferenceModel(modelLoader.loadInferenceModel(), classes.size());
      optimizerModel = new LiteOptimizerModel(modelLoader.loadOptimizerModel());
    } catch (IOException e) {
      throw new RuntimeException("Couldn't read underlying models for TransferLearningModel", e);
    }

    this.bottleneckShape = bottleneckModel.getBottleneckShape();
    int[] modelParameterSizes = trainHeadModel.getParameterSizes();

    modelParameters = new ByteBuffer[modelParameterSizes.length];
    modelGradients = new ByteBuffer[modelParameterSizes.length];
    nextModelParameters = new ByteBuffer[modelParameterSizes.length];

    for (int parameterIndex = 0; parameterIndex < modelParameterSizes.length; parameterIndex++) {
      int bufferSize = modelParameterSizes[parameterIndex] * FLOAT_BYTES;
      modelParameters[parameterIndex] = allocateBuffer(bufferSize);
      modelGradients[parameterIndex] = allocateBuffer(bufferSize);
      nextModelParameters[parameterIndex] = allocateBuffer(bufferSize);
    }
    initializeModel.initializeParameters(modelParameters);

    int[] optimizerStateElementSizes = optimizerModel.stateElementSizes();
    optimizerState = new ByteBuffer[optimizerStateElementSizes.length];
    nextOptimizerState = new ByteBuffer[optimizerStateElementSizes.length];

    for (int elemIdx = 0; elemIdx < optimizerState.length; elemIdx++) {
      int bufferSize = optimizerStateElementSizes[elemIdx] * FLOAT_BYTES;
      optimizerState[elemIdx] = allocateBuffer(bufferSize);
      nextOptimizerState[elemIdx] = allocateBuffer(bufferSize);
      fillBufferWithZeros(optimizerState[elemIdx]);
    }

    trainingBatchBottlenecks =
        allocateBuffer(getTrainBatchSize() * numBottleneckFeatures() * FLOAT_BYTES);

    int batchClassesNumElements = getTrainBatchSize() * classes.size();
    trainingBatchClasses = allocateBuffer(batchClassesNumElements * FLOAT_BYTES);
    zeroBatchClasses = allocateBuffer(batchClassesNumElements * FLOAT_BYTES);
    for (int idx = 0; idx < batchClassesNumElements; idx++) {
      zeroBatchClasses.putFloat(0);
    }
    zeroBatchClasses.rewind();

    inferenceBottleneck = allocateBuffer(numBottleneckFeatures() * FLOAT_BYTES);
  }

  /**
   * Adds a new sample for training.
   *
   * Sample bottleneck is generated in a background thread, which resolves the returned Future
   * when the bottleneck is added to training samples.
   *
   * @param image image RGB data.
   * @param className ground truth label for image.
   */
  public Future<Void> addSample(float[] image, String className) {
    checkNotTerminating();

    if (!classes.containsKey(className)) {
      throw new IllegalArgumentException(String.format(
          "Class \"%s\" is not one of the classes recognized by the model", className));
    }

    return executor.submit(() -> {
      ByteBuffer imageBuffer = allocateBuffer(image.length * FLOAT_BYTES);
      for (float f : image) {
        imageBuffer.putFloat(f);
      }
      imageBuffer.rewind();

      if (Thread.interrupted()) {
        return null;
      }
      ByteBuffer bottleneck = bottleneckModel.generateBottleneck(imageBuffer, null);

      trainingLock.lockInterruptibly();
      try {
        trainingSamples.add(new TrainingSample(bottleneck, className));
      } finally {
        trainingLock.unlock();
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

    if (trainingSamples.size() < getTrainBatchSize()) {
      throw new RuntimeException(
          String.format(
              "Too few samples to start training: need %d, got %d",
              getTrainBatchSize(), trainingSamples.size()));
    }

    return executor.submit(
        () -> {
          trainingLock.lock();
          try {
            epochLoop:
            for (int epoch = 0; epoch < numEpochs; epoch++) {
              float totalLoss = 0;
              int numBatchesProcessed = 0;

              for (List<TrainingSample> batch : trainingBatches()) {
                if (Thread.interrupted()) {
                  break epochLoop;
                }

                trainingBatchClasses.put(zeroBatchClasses);
                trainingBatchClasses.rewind();
                zeroBatchClasses.rewind();

                for (int sampleIdx = 0; sampleIdx < batch.size(); sampleIdx++) {
                  TrainingSample sample = batch.get(sampleIdx);
                  trainingBatchBottlenecks.put(sample.bottleneck);
                  sample.bottleneck.rewind();

                  // Fill trainingBatchClasses with one-hot.
                  int position =
                      (sampleIdx * classes.size() + classes.get(sample.className)) * FLOAT_BYTES;
                  trainingBatchClasses.putFloat(position, 1);
                }
                trainingBatchBottlenecks.rewind();

                float loss =
                    trainHeadModel.calculateGradients(
                        trainingBatchBottlenecks,
                        trainingBatchClasses,
                        modelParameters,
                        modelGradients);
                totalLoss += loss;
                numBatchesProcessed++;

                optimizerModel.performStep(
                    modelParameters,
                    modelGradients,
                    optimizerState,
                    nextModelParameters,
                    nextOptimizerState);

                ByteBuffer[] swapBufferArray;

                // Swap optimizer state with its next version.
                swapBufferArray = optimizerState;
                optimizerState = nextOptimizerState;
                nextOptimizerState = swapBufferArray;

                // Swap model parameters with their next versions.
                parameterLock.writeLock().lock();
                try {
                  swapBufferArray = modelParameters;
                  modelParameters = nextModelParameters;
                  nextModelParameters = swapBufferArray;
                } finally {
                  parameterLock.writeLock().unlock();
                }
              }

              float avgLoss = totalLoss / numBatchesProcessed;
              if (lossConsumer != null) {
                lossConsumer.onLoss(epoch, avgLoss);
              }
            }

            return null;
          } finally {
            trainingLock.unlock();
          }
        });
  }

  /**
   * Runs model inference on a given image.
   * @param image image RGB data.
   * @return predictions sorted by confidence decreasing. Can be null if model is terminating.
   */
  public Prediction[] predict(float[] image) {
    checkNotTerminating();
    inferenceLock.lock();

    try {
      if (isTerminating) {
        return null;
      }

      ByteBuffer imageBuffer = allocateBuffer(image.length * FLOAT_BYTES);
      for (float f : image) {
        imageBuffer.putFloat(f);
      }
      imageBuffer.rewind();

      ByteBuffer bottleneck = bottleneckModel.generateBottleneck(imageBuffer, inferenceBottleneck);

      float[] confidences;
      parameterLock.readLock().lock();
      try {
        confidences = inferenceModel.runInference(bottleneck, modelParameters);
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
      inferenceLock.unlock();
    }
  }

  /**
   * Writes the current values of the model parameters to a writable channel.
   *
   * The written values can be restored later using {@link #loadParameters(ScatteringByteChannel)},
   * under condition that the same underlying model is used.
   *
   * @param outputChannel where to write the parameters.
   * @throws IOException if an I/O error occurs.
   */
  public void saveParameters(GatheringByteChannel outputChannel) throws IOException {
    parameterLock.readLock().lock();
    try {
      outputChannel.write(modelParameters);
      for (ByteBuffer buffer : modelParameters) {
        buffer.rewind();
      }
    } finally {
      parameterLock.readLock().unlock();
    }
  }

  /**
   * Overwrites the current model parameter values with the values read from a channel.
   *
   * The channel should contain values previously written by
   * {@link #saveParameters(GatheringByteChannel)} for the same underlying model.
   *
   * @param inputChannel where to read the parameters from.
   * @throws IOException if an I/O error occurs.
   */
  public void loadParameters(ScatteringByteChannel inputChannel) throws IOException {
    parameterLock.writeLock().lock();
    try {
      inputChannel.read(modelParameters);
      for (ByteBuffer buffer : modelParameters) {
        buffer.rewind();
      }
    } finally {
      parameterLock.writeLock().unlock();
    }
  }

  /** Training model expected batch size. */
  public int getTrainBatchSize() {
    return trainHeadModel.getBatchSize();
  }

  /**
   * Constructs an iterator that iterates over training sample batches.
   * @return iterator over batches.
   */
  private Iterable<List<TrainingSample>> trainingBatches() {
    if (!trainingLock.tryLock()) {
      throw new RuntimeException("Thread calling trainingBatches() must hold the training lock");
    }
    trainingLock.unlock();

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
            int toIndex = nextIndex + getTrainBatchSize();
            nextIndex = toIndex;
            if (toIndex >= trainingSamples.size()) {
              // To keep batch size consistent, last batch may include some elements from the
              // next-to-last batch.
              return trainingSamples.subList(
                  trainingSamples.size() - getTrainBatchSize(), trainingSamples.size());
            } else {
              return trainingSamples.subList(fromIndex, toIndex);
            }
          }
        };
  }

  private void checkNotTerminating() {
    if (isTerminating) {
      throw new IllegalStateException("Cannot operate on terminating model");
    }
  }

  private int numBottleneckFeatures() {
    int result = 1;
    for (int size : bottleneckShape) {
      result *= size;
    }

    return result;
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
    inferenceLock.lock();

    try {
      boolean ok = executor.awaitTermination(5, TimeUnit.SECONDS);
      if (!ok) {
        throw new RuntimeException("Model thread pool failed to terminate");
      }

      initializeModel.close();
      bottleneckModel.close();
      trainHeadModel.close();
      inferenceModel.close();
      optimizerModel.close();
    } catch (InterruptedException e) {
      // no-op
    } finally {
      inferenceLock.unlock();
    }
  }

  private static ByteBuffer allocateBuffer(int capacity) {
    ByteBuffer buffer = ByteBuffer.allocateDirect(capacity);
    buffer.order(ByteOrder.nativeOrder());
    return buffer;
  }

  private static void fillBufferWithZeros(ByteBuffer buffer) {
    int bufSize = buffer.capacity();
    int chunkSize = Math.min(1024, bufSize);

    ByteBuffer zerosChunk = allocateBuffer(chunkSize);
    for (int idx = 0; idx < chunkSize; idx++) {
      zerosChunk.put((byte) 0);
    }
    zerosChunk.rewind();

    for (int chunkIdx = 0; chunkIdx < bufSize / chunkSize; chunkIdx++) {
      buffer.put(zerosChunk);
    }
    for (int idx = 0; idx < bufSize % chunkSize; idx++) {
      buffer.put((byte) 0);
    }
  }
}
